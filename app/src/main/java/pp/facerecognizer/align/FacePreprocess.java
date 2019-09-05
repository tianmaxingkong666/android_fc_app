package pp.facerecognizer.align;

import android.graphics.Bitmap;

public class FacePreprocess {
    static {
        try {
            System.loadLibrary("FacePreprocessSo");
            System.out.println("加载FacePreprocessSo成功");
        }catch (Exception e){
            System.out.println("加载FacePreprocessSo失败");
        }
    }

    public static native Bitmap facePreprocess(Bitmap bitmap, float[][] landmark);

}
