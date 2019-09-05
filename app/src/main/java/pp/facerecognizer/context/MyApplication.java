package pp.facerecognizer.context;

import android.app.Application;
import android.content.Context;


//import org.opencv.android.OpenCVLoader;

/**
 * Created by zhao on 2016/11/24.
 */
public class MyApplication extends Application {
    public static Context context;

    @Override
    public void onCreate() {
        super.onCreate();

        context = getApplicationContext();
    }

    public static Context getContext() {
        return context;
    }
}
