package com.larplarpsahur.pipeline;

import java.text.Normalizer;
import java.util.*;

/**
 * NFC/lowercase/punctuation-removal profile; corpus counts are aggregated, never averaged.
 * Translation quality is reported as BLEU (sacreBLEU 13a/zh), chrF2 (sacreBLEU defaults),
 * CER, character accuracy (1 - CER, floored at 0) and exact match after normalization.
 * A "+num" language suffix ("vi+num") also spells digits out, for reference tables that
 * were delivered normalized (Round 3 E_to_V: lowercase, no punctuation, numbers in words).
 */
public final class Metrics {
    public static final String PROFILE="nfc-lower-stripP-space-v1";
    private Metrics() {}
    public static final String NUMBERS="+num",NUMBER_PROFILE="spell-digits-v1";
    /** Language code for scoring a sample whose reference table is normalized. */
    public static String scoring(String lang,boolean normalizedReference) {
        return normalizedReference&&("vi".equals(lang)||"en".equals(lang))?lang+NUMBERS:lang;
    }
    private static String base(String lang) { return lang!=null&&lang.endsWith(NUMBERS)?lang.substring(0,lang.length()-NUMBERS.length()):lang; }
    public static String normalize(String text,String lang) {
        String s=Normalizer.normalize(text==null?"":text,Normalizer.Form.NFC).toLowerCase(Locale.ROOT);
        // Normalized tables carry no symbols either ("ten dollars", not "$10").
        if(lang!=null&&lang.endsWith(NUMBERS)) { lang=base(lang);s=Numbers.spell(s,lang).replaceAll("\\p{S}"," "); }
        s=s.replaceAll("\\p{P}","").replaceAll("[\\s\\p{Z}]+"," ").trim();
        return "zh".equals(lang)?s.replace(" ",""):s;
    }
    public static boolean exact(String ref,String hyp,String lang) {
        String r=normalize(ref,lang);return !r.isEmpty()&&r.equals(normalize(hyp,lang));
    }
    /** 100 * (1 - CER), floored at 0: the share of reference characters the hypothesis got right. */
    public static Double accuracy(Errors chars) { Double rate=chars.rate();return rate==null?null:100*Math.max(0,1-rate); }
    public static final class Errors {
        public long edits,length;
        public Errors(long e,long l) { edits=e;length=l; }
        public Double rate() { return length==0?null:edits/(double)length; }
        public void add(Errors e) { edits+=e.edits;length+=e.length; }
    }
    public static Errors chars(String ref,String hyp,String lang) {
        int[] r=normalize(ref,lang).codePoints().toArray(),h=normalize(hyp,lang).codePoints().toArray();
        if(r.length==0)return new Errors(0,0);
        return new Errors(distance(r,h),r.length);
    }
    public static Errors words(String ref,String hyp,String lang) {
        if("zh".equals(base(lang)))return new Errors(0,0);
        String r=normalize(ref,lang),h=normalize(hyp,lang);
        if(r.isEmpty())return new Errors(0,0);
        Map<String,Integer> ids=new HashMap<>();
        return new Errors(distance(wordIds(r,ids),wordIds(h,ids)),r.isEmpty()?0:r.split(" ").length);
    }
    private static int[] wordIds(String text,Map<String,Integer> ids) {
        if(text.isEmpty())return new int[0];String[] words=text.split(" ");int[] out=new int[words.length];
        for(int i=0;i<words.length;i++)out[i]=ids.computeIfAbsent(words[i],x->ids.size());return out;
    }
    static int distance(int[] a,int[] b) {
        if(a.length<b.length)return distance(b,a);
        int[] prev=new int[b.length+1],next=new int[b.length+1];for(int j=0;j<=b.length;j++)prev[j]=j;
        for(int i=1;i<=a.length;i++) {
            next[0]=i;for(int j=1;j<=b.length;j++) next[j]=Math.min(prev[j-1]+(a[i-1]==b[j-1]?0:1),Math.min(prev[j]+1,next[j-1]+1));
            int[] swap=prev;prev=next;next=swap;
        }
        return prev[b.length];
    }
    public static final class Bleu {
        public final long[] matches=new long[4],total=new long[4];
        public long refLength,hypLength;public int references;
        public void add(String ref,String hyp,String lang) {
            if(ref==null||normalize(ref,lang).isEmpty())return;
            references++;
            List<String> r=tokens(ref,lang),h=tokens(hyp,lang);refLength+=r.size();hypLength+=h.size();
            for(int n=1;n<=4;n++) {
                Map<List<String>,Integer> rc=ngrams(r,n),hc=ngrams(h,n);total[n-1]+=Math.max(0,h.size()-n+1);
                for(Map.Entry<List<String>,Integer> e:hc.entrySet()) matches[n-1]+=Math.min(e.getValue(),rc.getOrDefault(e.getKey(),0));
            }
        }
        public Double score(boolean effectiveOrder) {
            if(references==0)return null;if(hypLength==0)return 0.0;
            long sum=0;for(long m:matches)sum+=m;if(sum==0)return 0.0;
            double logs=0,smooth=1;int order=4;
            for(int i=0;i<4;i++) {
                if(total[i]==0) { if(effectiveOrder) { order=i;break; } return 0.0; }
                double precision;
                if(matches[i]==0) { smooth*=2;precision=1.0/(smooth*total[i]); }
                else precision=matches[i]/(double)total[i];
                logs+=Math.log(precision);
            }
            double bp=hypLength>=refLength?1:Math.exp(1-refLength/(double)hypLength);
            return order==0?0.0:100*bp*Math.exp(logs/order);
        }
    }
    private static Map<List<String>,Integer> ngrams(List<String> words,int n) {
        Map<List<String>,Integer> result=new HashMap<>();
        for(int i=0;i+n<=words.size();i++) {
            List<String> key=new ArrayList<>(words.subList(i,i+n));result.put(key,result.getOrDefault(key,0)+1);
        }return result;
    }
    /** sacreBLEU chrF defaults: character n-grams 1..6 without whitespace, beta 2, no word n-grams. */
    public static final class Chrf {
        private static final int ORDER=6;
        public final long[] hyp=new long[ORDER],ref=new long[ORDER],match=new long[ORDER];
        public int references;
        public void add(String reference,String hypothesis,String lang) {
            if(reference==null||normalize(reference,lang).isEmpty())return;
            references++;
            int[] r=normalize(reference,lang).replace(" ","").codePoints().toArray(),h=normalize(hypothesis,lang).replace(" ","").codePoints().toArray();
            for(int n=1;n<=ORDER;n++) {
                Map<String,Integer> rc=grams(r,n),hc=grams(h,n);
                // As sacreBLEU: hypothesis n-grams only count where the reference has n-grams of that order.
                if(r.length>=n)hyp[n-1]+=Math.max(0,h.length-n+1);
                ref[n-1]+=Math.max(0,r.length-n+1);
                for(Map.Entry<String,Integer> e:hc.entrySet())match[n-1]+=Math.min(e.getValue(),rc.getOrDefault(e.getKey(),0));
            }
        }
        public Double score() {
            if(references==0)return null;
            double precision=0,recall=0;int effective=0;
            for(int i=0;i<ORDER;i++)if(hyp[i]>0&&ref[i]>0) { precision+=match[i]/(double)hyp[i];recall+=match[i]/(double)ref[i];effective++; }
            if(effective==0)return 0.0;
            precision/=effective;recall/=effective;
            return precision+recall==0?0.0:100*5*precision*recall/(4*precision+recall);
        }
        private static Map<String,Integer> grams(int[] chars,int n) {
            Map<String,Integer> result=new HashMap<>();
            for(int i=0;i+n<=chars.length;i++)result.merge(new String(chars,i,n),1,Integer::sum);
            return result;
        }
    }
    public static List<String> tokens(String text,String lang) {
        String s=normalize(text,lang);lang=base(lang);
        if("zh".equals(lang)) {
            StringBuilder split=new StringBuilder();
            s.codePoints().forEach(cp->{ if(chinese(cp))split.append(' ').appendCodePoint(cp).append(' ');else split.appendCodePoint(cp); });s=split.toString();
        } else s=s.replace("<skipped>","").replace("-\n","").replace('\n',' ').replace("&quot;","\"").replace("&amp;","&").replace("&lt;","<").replace("&gt;",">");
        s=(" "+s+" ").replaceAll("([\\x7b-\\x7e\\x5b-\\x60\\x20-\\x26\\x28-\\x2b\\x3a-\\x40/])"," $1 ")
            .replaceAll("([^0-9])([.,])","$1 $2 ").replaceAll("([.,])([^0-9])"," $1 $2").replaceAll("([0-9])(-)","$1 $2 ")
            .replaceAll("[\\s\\p{Z}]+"," ").trim();
        return s.isEmpty()?Collections.emptyList():Arrays.asList(s.split(" "));
    }
    private static boolean chinese(int c) {
        int[][] ranges={{0x3400,0x4db5},{0x4e00,0x9fbb},{0xf900,0xfa2d},{0xfa30,0xfa6a},{0xfa70,0xfad9},
            {0x20000,0x2a6d6},{0x2f800,0x2fa1d},{0xff00,0xffef},{0x2e80,0x2eff},{0x3000,0x303f},{0x31c0,0x31ef},
            {0x2f00,0x2fdf},{0x2ff0,0x2fff},{0x3100,0x312f},{0x31a0,0x31bf},{0xfe10,0xfe1f},{0xfe30,0xfe4f},
            {0x2600,0x26ff},{0x2700,0x27bf},{0x3200,0x33ff}};
        for(int[] r:ranges)if(c>=r[0]&&c<=r[1])return true;return false;
    }
    public static String signature(String lang,boolean sentence) {
        return "nrefs:1|case:lc|eff:"+(sentence?"yes":"no")+"|tok:"+("zh".equals(base(lang))?"zh":"13a")+"|smooth:exp|profile:"+PROFILE
            +(lang!=null&&lang.endsWith(NUMBERS)?"+"+NUMBER_PROFILE:"")+"|implementation:larplarpsahur-1";
    }
    public static final String CHRF_SIGNATURE="nrefs:1|case:lc|eff:yes|nc:6|nw:0|space:no|implementation:larplarpsahur-1";
}
