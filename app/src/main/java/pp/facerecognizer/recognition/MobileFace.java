package pp.facerecognizer.recognition;

import android.graphics.Bitmap;


public class MobileFace {


    public static float[] getEmbeddings(Bitmap originalBitmap) {

        long startTime = System.currentTimeMillis();   //获取开始时间
        float[] emb = MxNetUtils.getFeatures(originalBitmap);
        long endTime=System.currentTimeMillis(); //获取结束时间

        System.out.println("人脸识别耗时： "+(endTime-startTime)+"ms");

        return emb;
    }

}
