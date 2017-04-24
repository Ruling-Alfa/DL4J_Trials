package com.alfa.dl4j_try;

import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.slf4j.Logger;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class mnistImagePipeLineNeuralNet {
	private static Logger log = LoggerFactory.getLogger(MNISTTrial1.class);
	
	public static void main(String[] args) throws IOException {
		long startTime = System.currentTimeMillis();
    	
		// image info
		// 28*28 grayscale
		
		int height = 28;
		int width = 28;
		int channels = 1;
		int rndSeed = 123;
		Random randNumber = new Random(rndSeed);
		int batchSize = 128;
		int outputNum = 10;
		int numEpocs = 15;
		
		File trainData = new File("D:\\Alfa\\mnistDataSet\\mnistDataSet");
		
		FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumber);
		
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		
		ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
		
		recordReader.initialize(train);
		
		DataSetIterator dataIttr = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
		
		DataNormalization scalar = new ImagePreProcessingScaler(0,1);
		scalar.fit(dataIttr);
		dataIttr.setPreProcessor(scalar);
		
		log.info("***** Building Model *****");
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(rndSeed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1)
				.learningRate(0.06)
				.updater(Updater.NESTEROVS).momentum(0.9)
				.regularization(true).l2(1e-4)
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(height*width)
						.nOut(100)
						.activation(Activation.RELU)
						.weightInit(WeightInit.XAVIER)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nIn(100)
						.nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.build())
				.pretrain(false).backprop(true)
				.setInputType(InputType.convolutional(height, width, channels))
				.build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		
		model.setListeners(new ScoreIterationListener(10));
		
		log.info("***** Training Model *****");
		
		for(int i = 0 ; i < numEpocs ; i++){
			model.fit(dataIttr);
		}
		
		log.info("Labels: " + recordReader.getLabels().toString());
		
		log.info("Dumping Model to a file....");
        final String saveName = "D:\\Alfa\\ML\\My_MNIST_Tarined_Model.zip";
        boolean saveUpdater = false;
        ModelSerializer.writeModel(model, saveName, saveUpdater);
        
        log.info("\nSaved to File: " + saveName);
        
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        log.info("Time elapsed: " + elapsedTime + " milli seconds");
        log.info("\nThat is " + elapsedTime/60000 + " minutes and " + (elapsedTime/1000)%60 + " seconds");
        log.info("\n****************Example finished********************");
	}

}