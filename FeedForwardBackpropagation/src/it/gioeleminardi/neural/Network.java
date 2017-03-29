package it.gioeleminardi.neural;

/**
 * Created by gioelem3 on 29/03/17.
 */
public class Network {

	/**
	 * The global error for the training
	 */
	protected double globalError;

	/**
	 * Number of the input neurons
	 */
	protected int inputCount;

	/**
	 * Number of the hidden layer neurons
	 */
	protected int hiddenCount;

	/**
	 * Number of the output neurons
	 */
	protected int outputCount;

	/**
	 * Total number of neurons in the network
	 */
	protected int neuronCount;

	/**
	 * The number of weight in the network
	 */
	protected int weightCount;

	/**
	 * The learning rate
	 */
	protected double learnRate;

	/**
	 * The output from the various levels
	 */
	protected double fire[];

	/**
	 * The weight matrix. It's the memory of the network
	 */
	protected double matrix[];

	/**
	 * The error from the last calculation
	 */
	protected double error[];

	/**
	 * Accumulates matrix's delta for training
	 */
	protected double accMatrixDelta[];

	/**
	 * The threholds. It's the memory of the network
	 */
	protected double thresholds[];

	/**
	 * The changes that should be applied to the weight matrix
	 */
	protected double matrixDelta[];

	/**
	 * The accumulation of thresholds deltas
	 */
	protected double accThresholdDelta[];

	/**
	 * The threshold deltas
	 */
	protected double thresholdDelta[];
}
