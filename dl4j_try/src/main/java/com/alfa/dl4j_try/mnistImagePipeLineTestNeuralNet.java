package com.alfa.dl4j_try;


import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFileChooser;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class mnistImagePipeLineTestNeuralNet {

    private static Logger log = LoggerFactory.getLogger(mnistImagePipeLineTestNeuralNet.class);

    public static String chooseFile(){
    	JFileChooser fc = new JFileChooser();
    	if(fc.showOpenDialog(null)==JFileChooser.APPROVE_OPTION){
    		File file = fc.getSelectedFile();
    		String filename = file.getAbsolutePath();
    		return filename;
    	}
    	
    	else return null;
    }
    
    public static void main(String[] args) throws Exception {
    	List<Integer> labels = Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    	String fileChose = chooseFile();
    	
    	long startTime = System.currentTimeMillis();
    	int height = 28;
		int width = 28;
		int channels = 1;

        log.info("Loading Model from file....");
        final String saveName = "D:\\Alfa\\ML\\My_MNIST_Tarined_Model.zip";
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(saveName);
       
        log.info("***** Testing your chosen file against the model");

        File file = new File(fileChose);
        
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        INDArray image = loader.asMatrix(file);
        DataNormalization scalar = new ImagePreProcessingScaler(0,1);
        scalar.transform(image);
        
        INDArray output = model.output(image);
        log.info("File Chosen was: " + fileChose);
        log.info("Probabilites: " + output.toString());
        log.info("Labels: " + labels.toString());
        
        Number maxProb = output.maxNumber();
        for(int i = 0 ; i < output.length() ; i ++){
        	if(Double.valueOf(output.getDouble(i)) == maxProb.doubleValue()){
        		log.info("Probably the image is of: " + labels.get(i).toString());
        		break;
        	}
        			
        }
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        log.info("Time elapsed: " + elapsedTime + " milli seconds");
        log.info("\nThat is " + elapsedTime/60000 + " minutes and " + (elapsedTime/1000)%60 + " seconds");
        log.info("\n****************Example finished********************");

    }

}
