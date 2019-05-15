package com.felmoreno1726.imageencyclopedia;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.InputStream;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    //PATH TO OUR MODEL FILE AND NAMES OF THE INPUT AND OUTPUT NODES
    private String MODEL_PATH = "file:///android_asset/output_model/mobile_model.pb";
    private String INPUT_NAME = "input_1";
    private String OUTPUT_NAME = "output_1";
    private TensorFlowInferenceInterface tf;

    //ARRAY TO HOLD THE PREDICTIONS AND FLOAT VALUES TO HOLD THE IMAGE DATA
    float[] PREDICTIONS = new float[1000];
    private float[] floatValues;
    private int[] INPUT_SIZE = {224,224,3};

    ImageView imageView;
    ImageView imageViewTest; //remove this
    TextView resultView;
    Snackbar progressBar;

    public Object[] argmax(float[] array) {
        /**
         * Computes the maximum predictions of confidence
         */
        int best = -1;
        float best_confidence = 0.0f;
        for (int i = 0; i < array.length; i++) {
            float value = array[i];
            if (value > best_confidence) {
                best_confidence = value;
                best = i;
            }
        }
        return new Object[]{best, best_confidence};
    }

    public void predict(final Bitmap bitmap){
        //Runs inference in background thread
        new AsyncTask<Integer,Integer,Integer>(){

            @Override
            protected Integer doInBackground(Integer ...params){
                //Resize the image into 224 x 224
                Bitmap resized_image = ImageUtils.processBitmap(bitmap,224);
                // remove everything beginning here:
                imageViewTest = (ImageView) findViewById(R.id.imageView2);
                imageViewTest.setImageBitmap(resized_image);
                // remove till here
                //Normalize the pixels
                floatValues = ImageUtils.normalizeBitmap(resized_image,224,127.5f,1.0f);
                //Pass input into the tensorflow
                tf.feed(INPUT_NAME,floatValues,1,224,224,3);
                //compute predictions
                tf.run(new String[]{OUTPUT_NAME});
                //copy the output into the PREDICTIONS array
                tf.fetch(OUTPUT_NAME,PREDICTIONS);
                System.out.println("Fetched predictions");
                //Obtained highest prediction
                Object[] results = argmax(PREDICTIONS);
                System.out.println("Predictions: " + Arrays.toString(PREDICTIONS));
                int class_index = (Integer) results[0];
                System.out.println("Class index: " + class_index);
                float confidence = (Float) results[1];
                System.out.println("Confident: " + confidence);
                try{
                    final String conf = String.valueOf(confidence * 100).substring(0,5);
                    System.out.println("Confidence: " + conf);
                    //Convert predicted class index into actual label name
                    //TODO
                    final String label = ImageUtils.getLabel(getAssets().open("labels.json"),class_index);//Needs a json file to go from val to label
                    System.out.println("Label: " + label);
                    //Display result on UI
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            progressBar.dismiss();
                            resultView.setText(label + " : " + conf + "%");
                        }
                    });
                }
                catch (Exception e){
                    System.out.println("Error when running prediction.");
                }
                return 0;
            }
        }.execute(0);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);


        tf = new TensorFlowInferenceInterface(getAssets(),MODEL_PATH);

        imageView = (ImageView) findViewById(R.id.imageView);
        resultView = (TextView) findViewById(R.id.TextViewResult);
        progressBar = Snackbar.make(imageView,"PROCESSING IMAGE",Snackbar.LENGTH_INDEFINITE);
        FloatingActionButton fab = findViewById(R.id.fab);//Currently used to initiate a predict
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try{
                    //READ THE IMAGE FROM ASSETS FOLDER
                    InputStream imageStream = getAssets().open("images/dog.jpeg");
                    Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
                    imageView.setImageBitmap(bitmap);
                    progressBar.show();
                    predict(bitmap);
                }
                catch (Exception e){
                    System.out.println("Error predicting image");
                }
            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}
