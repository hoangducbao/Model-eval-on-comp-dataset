package com.larplarpsahur.dataset;

import com.larplarpsahur.Io;
import org.xmlpull.v1.*;
import java.io.*;
import java.util.*;
import java.util.zip.*;

/** Bounded XLSX reader: shared/inline strings, cached numeric cells, sparse columns. */
public final class XlsxReader {
    private static XmlPullParser xml(ZipFile zip,String name) throws Exception {
        ZipEntry entry=zip.getEntry(name);
        if(entry==null) return null;
        String content;
        try(InputStream in=zip.getInputStream(entry)) { content=Io.read(in,32L*1024*1024); }
        if(content.contains("<!DOCTYPE")||content.contains("<!ENTITY")) throw new IOException("XML declarations are not allowed in XLSX");
        XmlPullParser p=XmlPullParserFactory.newInstance().newPullParser();p.setInput(new StringReader(content));return p;
    }
    public static List<Map<String,String>> read(File file) throws Exception {
        try(ZipFile zip=new ZipFile(file)) {
            List<String> strings=new ArrayList<>(); XmlPullParser p=xml(zip,"xl/sharedStrings.xml");
            if(p!=null) { StringBuilder s=null;
                for(int event=p.getEventType();event!=XmlPullParser.END_DOCUMENT;event=p.next()) {
                    if(event==XmlPullParser.START_TAG && "si".equals(p.getName())) s=new StringBuilder();
                    else if(event==XmlPullParser.START_TAG && "t".equals(p.getName()) && s!=null) s.append(p.nextText());
                    else if(event==XmlPullParser.END_TAG && "si".equals(p.getName())) { strings.add(s.toString());s=null; }
                }
            }
            String relationship=null,first=null;
            p=xml(zip,"xl/workbook.xml");
            if(p!=null) for(int e=p.getEventType();e!=XmlPullParser.END_DOCUMENT;e=p.next()) {
                if(e==XmlPullParser.START_TAG&&"sheet".equals(p.getName())) {
                    String id=p.getAttributeValue(null,"r:id"),name=p.getAttributeValue(null,"name");
                    if(first==null) first=id;
                    if("Samples".equalsIgnoreCase(name)||"Final".equalsIgnoreCase(name)) relationship=id;
                }
            }
            if(relationship==null) relationship=first;
            String sheet="xl/worksheets/sheet1.xml";
            p=xml(zip,"xl/_rels/workbook.xml.rels");
            if(p!=null) for(int e=p.getEventType();e!=XmlPullParser.END_DOCUMENT;e=p.next()) {
                if(e==XmlPullParser.START_TAG&&"Relationship".equals(p.getName())&&Objects.equals(relationship,p.getAttributeValue(null,"Id"))) {
                    String t=p.getAttributeValue(null,"Target");
                    sheet=t.startsWith("/")?t.substring(1):"xl/"+t;
                    if(sheet.contains("..")) throw new IOException("Unsafe worksheet path");
                }
            }
            p=xml(zip,sheet); if(p==null) throw new IOException("Worksheet missing");
            List<Map<String,String>> result=new ArrayList<>(); Map<Integer,String> headers=null,row=null;
            String type=null;int column=0;StringBuilder value=null;
            for(int e=p.getEventType();e!=XmlPullParser.END_DOCUMENT;e=p.next()) {
                if(e==XmlPullParser.START_TAG) {
                    String tag=p.getName();
                    if("row".equals(tag)) row=new TreeMap<>();
                    else if("c".equals(tag)) {
                        column=column(p.getAttributeValue(null,"r"));type=p.getAttributeValue(null,"t");value=new StringBuilder();
                    } else if(("v".equals(tag)||"t".equals(tag))&&value!=null) value.append(p.nextText());
                } else if(e==XmlPullParser.END_TAG) {
                    if("c".equals(p.getName())&&row!=null&&value!=null) {
                        String s=value.toString();
                        if("s".equals(type)&&!s.isEmpty()) s=strings.get(Integer.parseInt(s));
                        row.put(column,s.trim());value=null;
                    } else if("row".equals(p.getName())&&row!=null&&!row.isEmpty()) {
                        if(headers==null) headers=row;
                        else { Map<String,String> entry=new LinkedHashMap<>();for(Map.Entry<Integer,String> h:headers.entrySet()) entry.put(h.getValue().toLowerCase(Locale.ROOT),row.getOrDefault(h.getKey(),""));result.add(entry); }
                        if(result.size()>10000) throw new IOException("Too many XLSX rows");
                    }
                }
            }
            return result;
        }
    }
    private static int column(String r) throws IOException {
        if(r==null) throw new IOException("XLSX cell has no address");int c=0;
        for(char ch:r.toCharArray()) { if(ch<'A'||ch>'Z') break;c=c*26+ch-'A'+1; }
        if(c<1||c>256) throw new IOException("XLSX column outside supported range");return c;
    }
}
