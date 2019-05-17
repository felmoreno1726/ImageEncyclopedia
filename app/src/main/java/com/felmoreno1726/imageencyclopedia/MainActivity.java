package com.felmoreno1726.imageencyclopedia;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    //PATH TO OUR MODEL FILE AND NAMES OF THE INPUT AND OUTPUT NODES
    private final String MODEL_NAME = "mobilenet.pb";
    private final String MODEL_DIR = "file:///android_asset/tensorflow_models/";
    private String MODEL_PATH = MODEL_DIR + MODEL_NAME;
    private String INPUT_NAME = "input_1";
    private String OUTPUT_NAME = "output_1";
    private TensorFlowInferenceInterface tf;

    //ARRAY TO HOLD THE PREDICTIONS AND FLOAT VALUES TO HOLD THE IMAGE DATA
    float[] PREDICTIONS = new float[1000];
    private float[] floatValues;
    private int[] INPUT_SIZE = {224,224,3};

    ImageView imageView;
    TextView resultView;
    Snackbar progressBar;
    FloatingActionButton buttonLoadImage;
    FloatingActionButton buttonTakePhoto;
    private static int RESULT_LOAD_IMAGE = 1;
    private static int RESULT_TAKE_PICTURE = 2;

    public Object[] argmax(float[] array) {
    /*
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

        imageView = (ImageView) findViewById(R.id.imageView);
        resultView = (TextView) findViewById(R.id.TextViewResult);
        progressBar = Snackbar.make(imageView, "PROCESSING IMAGE", Snackbar.LENGTH_INDEFINITE);

        //

        buttonLoadImage = findViewById(R.id.galleryButton);
        buttonLoadImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Intent gallery = new Intent(Intent.ACTION_GET_CONTENT);
                //gallery.setType("image/*");
                //startActivityForResult(gallery, RESULT_LOAD_IMAGE);

                Intent photoPickEvent = new Intent(
                        Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(photoPickEvent, RESULT_LOAD_IMAGE);
                //Get image from activity callback TODO

            }
        });
        buttonTakePhoto = findViewById(R.id.photoButton);
        buttonTakePhoto.setOnClickListener(new View.OnClickListener(){
        URI output_URI;
            @Override
            public void onClick(View view){
                Intent takePhotoIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                //takePhotoIntent.putExtra(MediaStore.EXTRA_OUTPUT, uri);
                if (takePhotoIntent.resolveActivity(getPackageManager()) != null){
                    takePhotoIntent.putExtra(MediaStore.EXTRA_OUTPUT, output_URI);
                    startActivityForResult(takePhotoIntent, RESULT_TAKE_PICTURE);
                }
                startActivityForResult(takePhotoIntent, RESULT_TAKE_PICTURE);
                //finish();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        //If load image from
        Bitmap imageBitmap = null;
        Uri imageUri = null;
        if (resultCode == RESULT_OK) {
            if (requestCode == RESULT_LOAD_IMAGE) {
                imageUri = data.getData();
                //Bundle extras = data.getExtras();
                //Bitmap imageBitmap = (Bitmap) extras.get("data");
                System.out.println("Image from gallery has been uploaded");
                ImageView imageView = (ImageView) findViewById(R.id.imageView);
                imageView.setImageURI(imageUri);
                //Get the bitmap
                try {
                    imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            } else if (requestCode == RESULT_TAKE_PICTURE) {
                Bundle extras = data.getExtras();
                imageBitmap = (Bitmap) extras.get("data");
                data.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                System.out.println("Image has been captured");
                ImageView imageView = (ImageView) findViewById(R.id.imageView);
                System.out.println("This is the bitmap before setting imageview: " + imageBitmap);
                imageView.setImageBitmap(imageBitmap);
            }

            System.out.println("Bitmap has been set");
            //Then run the image on the model
            tf = new TensorFlowInferenceInterface(getAssets(), MODEL_PATH);
            try {
                //Bitmap imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                //READ THE IMAGE FROM ASSETS FOLDER
                InputStream imageStream = getAssets().open("images/dog.jpeg");
                Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
                //imageView.setImageBitmap(bitmap);
                progressBar.show();
                //wait(5);
                System.out.println("This is the bitmap: " + imageBitmap);
                predict(imageBitmap);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}