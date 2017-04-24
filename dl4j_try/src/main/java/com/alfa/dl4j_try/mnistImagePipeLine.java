package com.alfa.dl4j_try;

import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.slf4j.Logger;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class mnistImagePipeLine {
	private static Logger log = LoggerFactory.getLogger(MNISTTrial1.class);
	public static void main(String[] args) throws IOException {
		// image info
		// 28*28 grayscale
		
		int height = 28;
		int width = 28;
		int channels = 1;
		int rndSeed = 123;
		Random randNumber = new Random(rndSeed);
		int batchSize = 1;
		int outputNum = 10;
		
		File trainData = new File("D:\\Alfa\\mnistDataSet\\mnistDataSet");
		
		FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumber);
		
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		
		ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
		
		recordReader.initialize(train);
		recordReader.setListeners(new LogRecordListener());
		
		DataSetIterator dataIttr = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
		
		DataNormalization scalar = new ImagePreProcessingScaler(0,1);
		scalar.fit(dataIttr);
		
		for(int i = 0 ; i <3 ; i++){
			DataSet ds = dataIttr.next();
			System.out.println(ds);
			
			log.info(dataIttr.getLabels().toString());
		}
		

	}

}