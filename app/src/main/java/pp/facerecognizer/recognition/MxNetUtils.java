package pp.facerecognizer.recognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;

import org.dmlc.mxnet.Predictor;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhao on 2016/11/17.
 */
public class MxNetUtils {

    private MxNetUtils() {
    }

    public static float identifyImage(final Bitmap srcBitmap, final Bitmap dstBitmap) {

        float[] srcFeatures = getFeatures(srcBitmap);
        float[] dstFeatures = getFeatures(dstBitmap);

        float s = calCosineSimilarity(srcFeatures, dstFeatures);

        return s;
    }

    public static float[] getGrayArray(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(bitmap.getByteCount());
        bitmap.copyPixelsToBuffer(byteBuffer);
        byte[] bytes = byteBuffer.array();

        //float[] gray = new float[128 * 128];
        float[] gray = new float[3 * 112 * 112];

        for (int i = 0; i < bytes.length; i += 4) {
            int j = i / 4;
            float r = (float) (((int) (bytes[i + 0])) & 0xFF);
            float g = (float) (((int) (bytes[i + 1])) & 0xFF);
            float b = (float) (((int) (bytes[i + 2])) & 0xFF);

            //int temp = (int) (0.299 * r + 0.587 * g + 0.114 * b);
            //gray[j] = (float) (temp / 255.0);

//            gray[j] = (float) (r);
//            gray[j+1] = (float) (g);
//            gray[j+2] = (float) (b);

            gray[0 * 112 * 112 + j] = (float)(r);
            gray[1 * 112 * 112 + j] = (float)(g);
            gray[2 * 112 * 112 + j] = (float)(b);
        }
        return gray;
    }

    public static float[] do_flip(float[] arr) {
        for (int i = 0; i < arr.length / 2; i++) {
            float temp = arr[i];
            arr[i] = arr[arr.length - 1 - i];
            arr[arr.length - 1 - i] = temp;
        }
        return arr;
    }

    public static float[] normalize(float[] arr){
        float l2 = 0.f;
        for(float r : arr){
            l2 = l2 + (float)Math.pow(r,2);
        }
        l2 = (float)Math.sqrt(l2);
        for(int i=0;i<arr.length;i++){
            arr[i] = arr[i]/l2;
        }
        return arr;
    }

    public static float[] getFeatures(Bitmap srcBitmap) {
        float[] srcGray = getGrayArray(srcBitmap);
        long t = System.currentTimeMillis();
        Predictor predictor = FacePredictor.getPredictor();
        predictor.forward("data", srcGray);
        float[] result_org = predictor.getOutput(0);

        // 水平翻转再预测
        Matrix m = new Matrix();
        m.setScale(-1, 1);//水平翻转
        Bitmap reversePic = Bitmap.createBitmap(srcBitmap, 0, 0, srcBitmap.getWidth(), srcBitmap.getHeight(), m, true);
        float[] _srcGray = getGrayArray(reversePic);
        predictor.forward("data", _srcGray);
        float[] result_flip = predictor.getOutput(0);

        // 增加图片翻转后的编码
//        float[] _srcGray = do_flip(srcGray);

        float[] result = new float[128];
        for (int i = 0; i < result_org.length; i++){
            result[i] = result_org[i] + result_flip[i];
        }

        // 归一化
        float l2 = 0.f;
        for(float r : result){
            l2 = l2 + (float)Math.pow(r,2);
        }
        l2 = (float)Math.sqrt(l2);
        for(int i=0; i<result.length; i++){
            result[i] = result[i]/l2;
        }


        Log.d("verification time", String.valueOf(System.currentTimeMillis() - t));
        return result;
    }

    public static float calCosineSimilarity(float[] a, float[] b) {

        if (a.length != b.length) {
            return 0;
        }

        float n = 0;
        float x = 0;
        float y = 0;
        for (int i = 0; i < a.length; i++) {
            n += a[i] * b[i];
            x += a[i] * a[i];
            y += b[i] * b[i];
        }
        float s = (float) (n / (Math.sqrt(x) * Math.sqrt(y)));

        Log.d("main", "similarity" + s);

        return s;
    }



    public static int[] listToArray(List<String> list) {
        int[] arrayInt = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            int temp = Integer.parseInt(list.get(i));
            arrayInt[i] = temp;
        }
        return arrayInt;
    }

    public static List<String> readRawTextFile(Context ctx, int resId) {
        List<String> result = new ArrayList<>();
        InputStream inputStream = ctx.getResources().openRawResource(resId);

        InputStreamReader inputreader = new InputStreamReader(inputStream);
        BufferedReader buffreader = new BufferedReader(inputreader);
        String line;


        try {
            while ((line = buffreader.readLine()) != null) {
                result.add(line);
            }
        } catch (IOException e) {
            return null;
        }
        return result;
    }

}
