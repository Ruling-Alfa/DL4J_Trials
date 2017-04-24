package com.alfa.dl4j_try;


import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**A Simple Multi Layered Perceptron (MLP) applied to digit classification for
 * the MNIST Dataset (http://yann.lecun.com/exdb/mnist/).
 *
 * This file builds one input layer and one hidden layer.
 *
 * The input layer has input dimension of numRows*numColumns where these variables indicate the
 * number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
 * (relu) activation function. The weights for this layer are initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer will have 1000 output signals to the hidden layer.
 *
 * The hidden layer has input dimensions of 1000. These are fed from the input layer. The weights
 * for this layer is also initialized using Xavier initialization. The activation function for this
 * layer is a softmax, which normalizes all the 10 outputs such that the normalized sums
 * add up to 1. The highest of these normalized values is picked as the predicted class.
 *
 */
public class MNISTTrial2 {

    private static Logger log = LoggerFactory.getLogger(MNISTTrial2.class);

    public static void main(String[] args) throws Exception {
    	long startTime = System.currentTimeMillis();
    	
    	int outputNum = 10; // number of output classes
        int batchSize = 128; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility

        //Get the DataSetIterators:
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        log.info("Loading Model from file....");
        final String saveName = "D:\\Alfa\\ML\\MNIST_Tarined_Model.zip";
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(saveName);
       
        //System.out.println(model.conf().toString());
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        log.info("Time elapsed: " + elapsedTime + " milli seconds");
        log.info("\nThat is " + elapsedTime/60000 + " minutes and " + (elapsedTime/1000)%60 + " seconds");
        log.info("\n****************Example finished********************");

    }

}
