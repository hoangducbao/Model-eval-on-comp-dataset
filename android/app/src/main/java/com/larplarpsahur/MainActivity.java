package com.larplarpsahur;

import android.content.*;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.*;
import android.widget.*;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import java.util.*;

/** A compact live workspace for one ASR result and its translation. */
public final class MainActivity extends AppCompatActivity {
    private static final String[] LANGS={"vi","en","zh"};
    private EditText asrOutput,translationOutput;
    private TextView status,modelStatus,currentAudio,asrLanguage,translationLanguage;
    private Spinner target;
    private ProgressBar progress;
    private CheckBox trackTargets;
    private ArrayList<String> selected=new ArrayList<>();
    private String selectedNames="";
    private WorkService service;
    private boolean bound,busy;
    private WorkService.State latest;
    private final ActivityResultLauncher<String[]> pick=registerForActivityResult(
        new ActivityResultContracts.OpenMultipleDocuments(),uris->{
            if(uris.isEmpty())return;
            selected.clear();StringBuilder names=new StringBuilder();
            for(Uri uri:uris){
                try{getContentResolver().takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION);}catch(SecurityException ignored){}
                selected.add(uri.toString());
                if(names.length()>0)names.append('\n');
                names.append(service!=null?service.displayName(uri):uri.getLastPathSegment());
            }
            selectedNames=names.toString();showSelectedFile();startSelectedAudio();
        });
    private final ActivityResultLauncher<String> save=registerForActivityResult(
        new ActivityResultContracts.CreateDocument("application/zip"),uri->{
            if(uri!=null)start(new Intent(this,WorkService.class).setAction(WorkService.EXPORT).putExtra("destination",uri.toString()));
        });
    private final ServiceConnection connection=new ServiceConnection(){
        @Override public void onServiceConnected(ComponentName name,IBinder binder){
            service=((WorkService.LocalBinder)binder).service();service.observe(MainActivity.this::render);
        }
        @Override public void onServiceDisconnected(ComponentName name){service=null;}
    };
    @Override public void onCreate(Bundle state){
        super.onCreate(state);setContentView(R.layout.activity_main);
        asrOutput=findViewById(R.id.input_box);translationOutput=findViewById(R.id.output_box);status=findViewById(R.id.run_status);
        modelStatus=findViewById(R.id.model_status);progress=findViewById(R.id.run_progress);target=findViewById(R.id.target_language);
        currentAudio=findViewById(R.id.current_audio);asrLanguage=findViewById(R.id.asr_language);translationLanguage=findViewById(R.id.translation_language);
        trackTargets=findViewById(R.id.dataset_tracks);
        target.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"Tiếng Việt","English","中文"}));
        if(Build.VERSION.SDK_INT>=33&&checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[]{android.Manifest.permission.POST_NOTIFICATIONS},101);
        target.setSelection(getPreferences(MODE_PRIVATE).getInt("target",1));
        target.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener(){
            public void onNothingSelected(AdapterView<?> parent){}
            public void onItemSelected(AdapterView<?> parent,android.view.View view,int position,long id){
                getPreferences(MODE_PRIVATE).edit().putInt("target",position).apply();
                if(latest==null||latest.currentFile.isEmpty())
                    translationLanguage.setText("DỊCH · Ngôn ngữ đầu ra: "+language(LANGS[position]));
            }
        });
        if(state!=null){
            ArrayList<String> saved=state.getStringArrayList("uris");if(saved!=null)selected=saved;
            selectedNames=state.getString("names","");showSelectedFile();
        }
        findViewById(R.id.select_audio).setOnClickListener(v->pick.launch(new String[]{"audio/*","application/octet-stream"}));
        findViewById(R.id.retry_models).setOnClickListener(v->start(config(WorkService.INIT)));
        findViewById(R.id.cancel_work).setOnClickListener(v->{
            if(busy){if(service!=null)service.cancel();}
            else if(latest!=null&&latest.canResume)start(config(WorkService.RESUME));
        });
        findViewById(R.id.save_results).setOnClickListener(v->save.launch("larplarpsahur-results.zip"));
        findViewById(R.id.download_dataset).setOnClickListener(v->{
            selected.clear();selectedNames="";asrOutput.setText("");translationOutput.setText("");
            currentAudio.setText("Đang chuẩn bị quadata · ASR — · Dịch —");
            start(config(WorkService.DATASET).putExtra("track_targets",trackTargets.isChecked()));
        });
        updateButtons();
        if(state==null){
            if(!autoRun(getIntent()))start(config(WorkService.INIT));
        }
    }
    private Intent config(String action){
        getPreferences(MODE_PRIVATE).edit().putInt("target",target.getSelectedItemPosition()).apply();
        return new Intent(this,WorkService.class).setAction(action).putExtra("target_language",LANGS[target.getSelectedItemPosition()]);
    }
    private void start(Intent job){
        if(busy)return;
        try{busy=true;updateButtons();ContextCompat.startForegroundService(this,job);}
        catch(Exception e){busy=false;status.setText("Không bắt đầu được: "+Io.error(e));updateButtons();}
    }
    private void startSelectedAudio(){
        if(selected.isEmpty()){status.setText("Hãy chọn audio trước.");return;}
        asrOutput.setText("");translationOutput.setText("");
        start(config(WorkService.RUN).putStringArrayListExtra("uris",new ArrayList<>(selected)));
    }
    private boolean autoRun(Intent incoming){
        if(!incoming.getBooleanExtra("auto_run",false))return false;
        String lang=incoming.getStringExtra("target_language");
        for(int i=0;i<LANGS.length;i++)if(LANGS[i].equals(lang))target.setSelection(i);
        Intent job=config(WorkService.LOCAL);
        for(String key:new String[]{"dataset_root","mt_mode","tracks","out_dir"})if(incoming.hasExtra(key))job.putExtra(key,incoming.getStringExtra(key));
        job.putExtra("limit",incoming.getIntExtra("limit",0));
        job.putExtra("track_targets",incoming.getBooleanExtra("track_targets",true));
        start(job);return true;
    }
    @Override protected void onNewIntent(Intent intent){super.onNewIntent(intent);setIntent(intent);autoRun(intent);}
    @Override protected void onStart(){super.onStart();bound=bindService(new Intent(this,WorkService.class),connection,BIND_AUTO_CREATE);}
    @Override protected void onStop(){
        if(service!=null)service.observe(null);
        if(bound){unbindService(connection);bound=false;}service=null;super.onStop();
    }
    @Override protected void onSaveInstanceState(Bundle state){
        state.putStringArrayList("uris",selected);state.putString("names",selectedNames);super.onSaveInstanceState(state);
    }
    private void render(WorkService.State state){
        latest=state;busy=state.busy;status.setText(state.message);
        asrOutput.setText(state.asrText);translationOutput.setText(state.translationText);
        asrLanguage.setText("ASR · Ngôn ngữ phát hiện: "+language(state.sourceLanguage));
        translationLanguage.setText("DỊCH · Ngôn ngữ đầu ra: "+language(state.targetLanguage));
        renderCurrentAudio(state);
        modelStatus.setText(state.ready?"Model đã kiểm tra SHA-256 và sẵn sàng.":"Model chưa sẵn sàng. Cần Internet và thiết bị SM8850 cho lần thiết lập đầu.");
        progress.setIndeterminate(state.percent<0);if(state.percent>=0)progress.setProgress(state.percent);
        ((TextView)findViewById(R.id.language_hint)).setText(state.busy&&state.usingTrackTargets
            ?"Dataset: ngôn ngữ dịch theo từng track → "+language(state.targetLanguage)
            :"Chọn ngôn ngữ dịch cho Upfile ▾ · Dataset dùng hướng dịch từng track");
        updateButtons();
    }
    private void updateButtons(){
        for(int id:new int[]{R.id.select_audio,R.id.retry_models,R.id.download_dataset})findViewById(id).setEnabled(!busy);
        Button pause=findViewById(R.id.cancel_work);
        pause.setText(busy?(latest!=null&&latest.pauseRequested?"Đang dừng…":"Tạm dừng"):
            latest!=null&&latest.canResume?"Tiếp tục":"Tạm dừng");
        pause.setEnabled(latest!=null&&(busy?latest.canPause&&service!=null:latest.canResume));
        target.setEnabled(!busy);
        findViewById(R.id.save_results).setEnabled(!busy&&latest!=null&&latest.exportReady);
    }
    private void showSelectedFile(){
        if(selectedNames.isEmpty())return;
        String label=selectedNames.replace('\n',' ');
        if(label.length()>72)label=label.substring(0,69)+"…";
        currentAudio.setText(label+" · ASR — · Dịch —");
    }
    private void renderCurrentAudio(WorkService.State state){
        if(state.currentFile==null||state.currentFile.isEmpty()) { showSelectedFile();return; }
        currentAudio.setText(state.currentFile+" · ASR "+latency(state.asrMillis,state.busy&&"asr".equals(state.stage))+" · Dịch "+latency(state.translationMillis,state.busy&&"translation".equals(state.stage)));
    }
    private static String latency(long millis,boolean active){return millis<0?(active?"đang xử lý":"chưa có"):millis+" ms";}
    private static String language(String code){
        if("vi".equals(code))return "Tiếng Việt";
        if("en".equals(code))return "English";
        if("zh".equals(code))return "中文";
        if("und".equals(code))return "đang nhận diện";
        return code==null||code.isEmpty()?"—":code;
    }
}
