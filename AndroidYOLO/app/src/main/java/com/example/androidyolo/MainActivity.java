package com.example.androidyolo;

import android.Manifest;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.AudioManager;
import android.media.ToneGenerator;
import android.os.Environment;
import android.os.Handler;
import android.support.v4.app.ActivityCompat;
import android.support.v4.view.GestureDetectorCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
import static java.lang.Math.round;
import static org.opencv.core.Core.FILLED;
import static org.opencv.core.Core.FONT_HERSHEY_SIMPLEX;
import static org.opencv.imgproc.Imgproc.putText;
import static org.opencv.imgproc.Imgproc.rectangle;
import static org.opencv.imgproc.Imgproc.line;

import android.speech.tts.TextToSpeech;
import android.speech.tts.TextToSpeech.OnInitListener;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2,GestureDetector.OnGestureListener,GestureDetector.OnDoubleTapListener{
    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    private ArrayList<String>    classes = new ArrayList<>();
    String classesFile = "coco.names";
    String YoloCfg = "/Download/tiny.cfg";
    String YoloWeights = "/Download/tiny.weights";
    Map<String, String> chtw = new HashMap<String, String>() {{ put("person","人");put("bicycle","腳踏車");put("car","汽車");put("motorbike","摩托車");put("aeroplane","飛機");put("bus","公車");put("train","火車");put("truck","卡車");put("boat","船");put("traffic light","紅綠燈");put("fire hydrant","消防栓");put("stop sign","停止標誌");put("parking meter","停車收費表");put("bench","長凳");put("bird","鳥");put("cat","貓");put("dog","狗");put("horse","馬");put("sheep","羊");put("cow","牛");put("elephant","大象");put("bear","熊");put("zebra","斑馬");put("giraffe","長頸鹿");put("backpack","書包");put("umbrella","雨傘");put("handbag","手提包");put("tie","領帶");put("suitcase","手提箱");put("frisbee","飛盤");put("skis","滑雪板");put("snowboard","滑雪單板");put("sports ball","球類運動");put("kite","風箏");put("baseball bat","棒球棒");put("baseball glove","棒球手套");put("skateboard","滑板");put("surfboard","衝浪板");put("tennis racket","網球拍");put("bottle","瓶子");put("wine glass","酒杯");put("cup","杯子");put("fork","叉子");put("knife","刀子");put("spoon","湯匙");put("bowl","碗");put("banana","香蕉");put("apple","蘋果");put("sandwich","三明治");put("orange","柳橙");put("broccoli","花椰菜");put("carrot","胡蘿蔔");put("hot dog","熱狗");put("pizza","披薩");put("donut","甜甜圈");put("cake","蛋糕");put("chair","椅子");put("sofa","沙發");put("pottedplant","盆栽");put("bed","床");put("diningtable","餐桌");put("toilet","廁所");put("tvmonitor","螢幕");put("laptop","筆記型電腦");put("mouse","滑鼠");put("remote","遙控器");put("keyboard","鍵盤");put("cell phone","手機");put("microwave","微波爐");put("oven","烤箱");put("toaster","烤麵包機");put("sink","水槽");put("refrigerator","冰箱");put("book","書");put("clock","時鐘");put("vase","花瓶");put("scissors","剪刀");put("teddy bear","泰迪熊");put("hair drier","吹風機");put("toothbrush","牙刷");put("greenp","小綠人");put("redp","小紅人");put("tree","樹");put("transbox","變電箱");put("cones","三角錐");put("zebracross","斑馬線");}};
    float confThreshold = 0.4f;
    float nmsThreshold = 0.5f;
    int inpWidth = 256;
    int inpHeight = 256;
    Net net;
    private TextToSpeech tts; //語音
    boolean start = false; //開始/停止
    ToneGenerator alert; //警示聲
    boolean mode = true; //模式
    String light = "";
    boolean go = false;
    private GestureDetectorCompat mDetector; //手勢操作
    int CurrentSpeak = 0;
    boolean again = false;
    private Long startTime;
    private Handler handler = new Handler();
    float volume;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        createLanguageTTS();
        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        mDetector = new GestureDetectorCompat(this,this);
        startTime = System.currentTimeMillis();
        handler.removeCallbacks(updateTimer);
        handler.postDelayed(updateTimer, 1000);
        alert = new ToneGenerator(AudioManager.STREAM_ALARM, ToneGenerator.MAX_VOLUME);
        //調整亮度
//        Window localWindow = getWindow();
//        WindowManager.LayoutParams localLayoutParams = localWindow.getAttributes();
//        localLayoutParams.screenBrightness = 1.0f / 255.0f;
//        localWindow.setAttributes(localLayoutParams);
        if(ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED || ActivityCompat.checkSelfPermission(this, WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.CAMERA,WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE},1);
        }
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);
                if (status == BaseLoaderCallback.SUCCESS) {
                    cameraBridgeViewBase.enableView();

                    YoloCfg = Environment.getExternalStorageDirectory() + YoloCfg;
                    YoloWeights = Environment.getExternalStorageDirectory() + YoloWeights;
                    readClasses(classes, classesFile);
                    net = Dnn.readNetFromDarknet(YoloCfg, YoloWeights);
                    net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
                    net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
                } else {
                    super.onManagerConnected(status);
                }
            }
        };
    }

    public void Click(View view) {
        tts.speak( "長按螢幕開始或暫停，左右滑動切換模式，上下滑動調整語音速度", TextToSpeech.QUEUE_FLUSH, null );
    }

    @Override
    public boolean onTouchEvent(MotionEvent event)
    {
        this.mDetector.onTouchEvent(event);
        return super.onTouchEvent(event);
    }

    @Override
    public boolean onDown(MotionEvent e) {
        return false;
    }

    @Override
    public void onShowPress(MotionEvent e) { }

    @Override
    public boolean onSingleTapUp(MotionEvent e) {
        return false;
    }

    @Override
    public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {
        float uservol = getSharedPreferences("prefdata", MODE_PRIVATE)
                .getFloat("USER", volume);
        float sensitvity = 400;
        if((e1.getY() - e2.getY()) > sensitvity && distanceX <50){
            volume += 0.5;
        }else if((e2.getY() - e1.getY()) > sensitvity && distanceX <50){
            volume -= 0.5;
        }
        SharedPreferences pref = getSharedPreferences("prefdata", MODE_PRIVATE);
        pref.edit().putFloat("USER", volume);
        pref.edit().commit();

        tts.setSpeechRate(uservol); //速度
        tts.speak( "現在速度", TextToSpeech.QUEUE_FLUSH, null );
        return true;
    }

    @Override
    public void onLongPress(MotionEvent e) {
        try {
            if (!start){
                start = true;
                tts.speak( "開始識別", TextToSpeech.QUEUE_FLUSH, null );
                Toast.makeText(MainActivity.this, "開始識別", Toast.LENGTH_SHORT).show();
                findViewById(R.id.button).setVisibility(View.INVISIBLE);
            }
            else{
                start = false;
                tts.speak( "暫停識別", TextToSpeech.QUEUE_FLUSH, null );
                Toast.makeText(MainActivity.this, "暫停識別", Toast.LENGTH_SHORT).show();
                findViewById(R.id.button).setVisibility(View.VISIBLE);
            }
            TimeUnit.SECONDS.sleep(1);
        } catch (InterruptedException ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        float sensitvity = 300;
        try {
            if((e1.getX() - e2.getX()) > sensitvity){
                mode = true;
                tts.speak( "空間建構", TextToSpeech.QUEUE_FLUSH, null );
                Toast.makeText(MainActivity.this, "空間建構", Toast.LENGTH_SHORT).show();
            }else if((e2.getX() - e1.getX()) > sensitvity){
                mode = false;
                tts.speak( "閃避障礙，雙點擊念出障礙物", TextToSpeech.QUEUE_FLUSH, null );
                Toast.makeText(MainActivity.this, "閃避障礙", Toast.LENGTH_SHORT).show();
            }
            TimeUnit.SECONDS.sleep(1);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return mode;
    }

    @Override
    public boolean onSingleTapConfirmed(MotionEvent e) {
        return false;
    }

    @Override
    public boolean onDoubleTap(MotionEvent e) {
        if (!again){
            again = true;
            Toast.makeText(MainActivity.this, "重複", Toast.LENGTH_SHORT).show();
        }
        return again;
    }

    @Override
    public boolean onDoubleTapEvent(MotionEvent e) {
        return false;
    }

    private Runnable updateTimer = new Runnable() {
        public void run() {
            handler.postDelayed(this, 1000);
            Long spentTime = System.currentTimeMillis() - startTime;
            Long seconds = (spentTime/1000) % 60;
            if ((seconds%5) == 0){ //計時
                light = "";
            }
        }
    };

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();

        Mat dst = new Mat();
        line(frame,new Point(frame.width()/4*1.5,0),new Point(frame.width()/4*1.5,frame.height()),new Scalar(0, 255, 0),1);
        line(frame,new Point(frame.width()/4*2.5,0),new Point(frame.width()/4*2.5,frame.height()),new Scalar(0, 255, 0),1);
        line(frame,new Point(0,frame.height()/3*2),new Point(frame.width(),frame.height()/3*2),new Scalar(0, 255, 0),1);

        Imgproc.cvtColor(frame, dst, Imgproc.COLOR_BGRA2BGR);
        Mat blob = Dnn.blobFromImage(dst, 1 / 255.0, new Size(inpWidth, inpHeight), new Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        List<Mat> outs = new ArrayList<>();
        net.forward(outs, getOutputsNames(net));

        if (start) {
            postprocess(frame, outs);
        }
        return frame;
    }

    private void readClasses(ArrayList<String> classes, String file){
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(getAssets().open(file)));

            String mLine;
            while ((mLine = reader.readLine()) != null) {
                classes.add(mLine);
            }
        } catch (IOException e) {
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                }
            }
        }
    }

    List<String> getOutputsNames(Net net)
    {
        ArrayList<String> names = new ArrayList<>();
        if (names.size() == 0)
        {
            List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
            List<String> layersNames = net.getLayerNames();
            for (int i = 0; i < outLayers.size(); ++i) {
                String layer = layersNames.get(outLayers.get(i).intValue()-1);
                names.add(layer);
            }
        }
        return names;
    }

    private void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat frame)
    {
        rectangle(frame, new Point(left, top), new Point(right, bottom), new Scalar(255, 178, 50), 3);

        String label = String.format("%.2f", conf);
        if (classes.size() > 0)
        {
            label = classes.get(classId) + ":" + label;
        }
        int[] baseLine = new int[1];
        Size labelSize = Imgproc.getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
        top = java.lang.Math.max(top, (int)labelSize.height);
        rectangle(frame, new Point(left, top - round(1.5*labelSize.height)),
                new Point(left + round(1.5*labelSize.width), top + baseLine[0]), new Scalar(255, 255, 255), FILLED);
        putText(frame, label, new Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, new Scalar(0,0,0),1);
    }

    void postprocess(Mat frame, List<Mat> outs)
    {
        List<Integer> classIds = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Rect> boxes = new ArrayList<>();
        List<Float> objconf = new ArrayList<>();
        List<Integer> leftlabel = new ArrayList<>();
        List<Integer> rightlabel = new ArrayList<>();
        List<Integer>  centerlabel= new ArrayList<>();
        List<String>  item= new ArrayList<>();
        String zc = "";
        String gp = "";
        String rp = "";
        String say = "";
        float conf = 0;
        for (int i = 0; i < outs.size(); ++i)
        {
            for (int j = 0; j < outs.get(i).rows(); ++j)
            {
                Mat scores = outs.get(i).row(j).colRange(5, outs.get(i).row(j).cols());
                Core.MinMaxLocResult r = Core.minMaxLoc(scores);
                if (r.maxVal > confThreshold)
                {
                    Mat bb = outs.get(i).row(j).colRange(0, 5);
                    float[] data = new float[1];
                    bb.get(0, 0, data);
                    int centerX = (int)(data[0] * frame.cols());

                    bb.get(0, 1, data);
                    int centerY = (int)(data[0] * frame.rows());

                    bb.get(0, 2, data);
                    int width = (int)(data[0] * frame.cols());

                    bb.get(0, 3, data);
                    int height = (int)(data[0] * frame.rows());

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    bb.get(0, 4, data);
                    objconf.add(data[0]);

                    confidences.add((float)r.maxVal);
                    classIds.add((int)r.maxLoc.x);
                    boxes.add(new Rect(left, top, width, height));
                }
            }
        }
        MatOfRect boxs =  new MatOfRect();
        boxs.fromList(boxes);
        MatOfFloat confis = new MatOfFloat();
        confis.fromList(objconf);
        MatOfInt idxs = new MatOfInt();
        Dnn.NMSBoxes(boxs, confis, confThreshold, nmsThreshold, idxs);
        if(idxs.total() > 0) {
            int[] indices = idxs.toArray();
            for (int idx : indices) {
                Rect box = boxes.get(idx);
                drawPred(classIds.get(idx), confidences.get(idx), box.x, box.y,
                        box.x + box.width, box.y + box.height, frame);
                int x = (box.x + (box.x + box.width)) / 2;
                if ((x > ((frame.width() / 4) * 1.5)) && (x < ((frame.width() / 4) * 2.5))) {
                    String label = classes.get(classIds.get(idx));
                    if ((box.y + box.height) > (frame.height() / 3 * 2)) {
                        item.add(chtw.get(label));
                        if(confidences.get(idx) > conf){
                            conf = confidences.get(idx);
                            say = chtw.get(classes.get(classIds.get(idx)));
                        }else {
                            say = chtw.get(classes.get(classIds.get(idx)));
                        }
                        centerlabel.add(x);
                    } else {
                        item.add(chtw.get(label));
                    }
                    switch (label) {
                        case "greenp":
                            gp = label;
                            if (light.equals(label)) {
                                go = false;
                                light = label;
                            } else {
                                go = true;
                                light = label;
                            }
                            break;
                        case "redp":
                            rp = label;
                            if (light.equals(label)) {
                                go = false;
                                light = label;
                            } else {
                                go = true;
                                light = label;
                            }
                            break;
                        case "zebracross":
                            zc = label;
                            break;
                    }
                } else if (x > ((frame.width() / 4) * 2.5)) {
                    rightlabel.add(x);
                } else if (x < ((frame.width() / 4) * 1.5)) {
                    leftlabel.add(x);
                }
            }
        }
        else {
            CurrentSpeak = 0;
        }
        if (mode){
            if (classes.size() > 0)
            {
                for (int i=0;i<item.size();i++){
                    try {
                        tts.speak(item.get(i), TextToSpeech.QUEUE_FLUSH, null );
                        TimeUnit.SECONDS.sleep(1);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        else {
            if(light.length() == 0){
                if(!centerlabel.isEmpty()){
                    if (rightlabel.isEmpty()){
                        if (CurrentSpeak != 1){
                            try {
                                alert.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 1000);
                                tts.speak( "前方有障礙物，請靠右", TextToSpeech.QUEUE_FLUSH, null );
                                TimeUnit.SECONDS.sleep(2);
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                            CurrentSpeak = 1;
                        }
                    }
                    else if(leftlabel.isEmpty()){
                        if (CurrentSpeak != 2){
                            try {
                                alert.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 1000);
                                tts.speak( "前方有障礙物，請靠左", TextToSpeech.QUEUE_FLUSH, null );
                                TimeUnit.SECONDS.sleep(2);
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                            CurrentSpeak = 2;
                        }
                    }
                    else {
                        if (CurrentSpeak != 3){
                            try {
                                alert.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 1000);
                                TimeUnit.SECONDS.sleep(1);
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                            try {
                                alert.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 1000);
                                TimeUnit.SECONDS.sleep(1);
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                            CurrentSpeak = 3;
                        }
                    }
                    if (again){
                        try {
                            tts.speak( say, TextToSpeech.QUEUE_FLUSH, null );
                            TimeUnit.SECONDS.sleep(1);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                        again = false;
                    }
                }
                else{
                    CurrentSpeak = 0;
                }
            }
            else {
                if (go){
                    if(zc.length() > 0 && gp.length() > 0){
                        try {
                            tts.speak( "前方綠燈", TextToSpeech.QUEUE_FLUSH, null );
                            TimeUnit.SECONDS.sleep(2);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    else if(zc.length() > 0 && rp.length() > 0){
                        try {
                            tts.speak( "前方紅燈", TextToSpeech.QUEUE_FLUSH, null );
                            TimeUnit.SECONDS.sleep(2);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    else {
                        try {
                            tts.speak( "前方路口", TextToSpeech.QUEUE_FLUSH, null );
                            TimeUnit.SECONDS.sleep(2);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
                else {
                    if (rp.length() > 0){
                        try {
                            tts.speak( "前方紅燈", TextToSpeech.QUEUE_FLUSH, null );
                            TimeUnit.SECONDS.sleep(2);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }
    }

    private void createLanguageTTS()
    {
        if( tts == null )
        {
            tts = new TextToSpeech(this, new OnInitListener(){
                @Override
                public void onInit(int arg0)
                {
                    if( arg0 == TextToSpeech.SUCCESS )
                    {
                        Locale l = Locale.TAIWAN;

                        if( tts.isLanguageAvailable( l ) == TextToSpeech.LANG_COUNTRY_AVAILABLE )
                        {
                            tts.setLanguage(l);
                            tts.setPitch(0);    //語調
                        }
                    }
                }}
            );
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) { }

    @Override
    public void onCameraViewStopped() { }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"ERROR", Toast.LENGTH_SHORT).show();
        }
        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraBridgeViewBase.disableView();
        tts.shutdown();
    }
}