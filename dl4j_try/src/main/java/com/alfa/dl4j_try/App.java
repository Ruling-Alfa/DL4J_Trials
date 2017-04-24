package com.alfa.dl4j_try;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;


/**
 * Hello world!
 *
 */
public class App 
{
    public static void main(String[] args )
    {
        System.out.println( "Hello World!" );
        
        try {
			DataSetIterator trainng = new MnistDataSetIterator(128, true, 123);
			DataSetIterator testing = new MnistDataSetIterator(128, false, 123);
			
			
			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.seed(1)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.iterations(1)
					.learningRate(0.5)
					.updater(Updater.NESTEROVS).momentum(0.2)
					.regularization(true).l2(1e-2)
					.list()
					.layer(0, new DenseLayer.Builder()
							.nIn(28*28)
							.nOut(5000)
							.activation(Activation.RELU)
							.weightInit(WeightInit.XAVIER)
							.build()
					)
					.layer(1, new OutputLayer.Builder(org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
							.nIn(5000)
							.nOut(10)
							.activation(Activation.SOFTMAX)
							.weightInit(WeightInit.XAVIER)
							.build()
					)
					.pretrain(false)
					.backprop(true)
					.build();
			
			MultiLayerNetwork network = new MultiLayerNetwork(conf);
			network.init();
			network.setListeners(new ScoreIterationListener(1));
			
			System.out.println("Train model....");
	        for( int i=0; i<15; i++ ){
	            network.fit(trainng);
	        }
	        
	        System.out.println("Evaluate model....");

	        org.deeplearning4j.eval.Evaluation eval = new org.deeplearning4j.eval.Evaluation(10);
	        while(testing.hasNext()){
	        	org.nd4j.linalg.dataset.DataSet next = testing.next();
	        	INDArray out = network.output(next.getFeatureMatrix());
	        	eval.eval(next.getLabels(), out);
	        }
	       System.out.println(eval.stats());
	        System.out.println("\n****************Example finished********************");
			
		} catch (IOException e) {
			e.printStackTrace();
		}
        
    }
}
