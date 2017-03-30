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
	 * The momentum of the training
	 */
	protected double momentum;

	/**
	 * The output from the various levels
	 */
	protected double fire[];

	/**
	 * The weight matrix. It's the memory of the network
	 */
	protected double matrix[];

	/**
	 * The changes that should be applied to the weight matrix
	 */
	protected double matrixDelta[];

	/**
	 * Accumulates matrix's delta for training
	 */
	protected double accMatrixDelta[];

	/**
	 * The threholds. It's the memory of the network
	 */
	protected double thresholds[];

	/**
	 * The threshold deltas
	 */
	protected double thresholdDelta[];

	/**
	 * The accumulation of thresholds deltas
	 */
	protected double accThresholdDelta[];

	/**
	 * The error from the last calculation
	 */
	protected double error[];

	/**
	 * The changes in the errors
	 */
	protected double errorDelta[];

	/**
	 * The constructor of the Network
	 *
	 * @param inputCount  The number of input neurons
	 * @param hiddenCount The number of hidden neurons
	 * @param outputCount The number of output neurons
	 * @param learnRate   The learn rate used in the training
	 * @param momentum    The momentum used in the training
	 */
	public Network(int inputCount, int hiddenCount, int outputCount, double learnRate, double momentum) {
		this.inputCount = inputCount;
		this.hiddenCount = hiddenCount;
		this.outputCount = outputCount;
		this.learnRate = learnRate;
		this.momentum = momentum;

		this.neuronCount = inputCount + hiddenCount + outputCount;
		this.weightCount = (inputCount * hiddenCount) + (hiddenCount * outputCount);

		this.fire = new double[neuronCount];
		this.matrix = new double[weightCount];
		this.matrixDelta = new double[weightCount];
		this.accMatrixDelta = new double[weightCount];
		this.thresholds = new double[neuronCount];
		this.thresholdDelta = new double[neuronCount];
		this.accThresholdDelta = new double[neuronCount];
		this.error = new double[neuronCount];
		this.errorDelta = new double[neuronCount];

		reset();
	}

	/**
	 * Activation function.
	 * In this case is a Sigmoid Function
	 *
	 * @param sum The activation from the neuron
	 * @return The activation applied to the threshold function
	 */
	public double threshold(double sum) {
		return 1.0 / (1 + Math.exp(-1.0 * sum));
	}

	/**
	 * Compute the output for a given input to the network.
	 *
	 * @param input The input of the neural net
	 * @return The results from the output neurons
	 */
	public double[] computeOutputs(double input[]) {
		int i, j;
		final int hiddenIndex = hiddenCount;
		final int outIndex = inputCount + hiddenCount;

		for (i = 0; i < inputCount; i++) {
			fire[i] = input[i];
		}

		int inx = 0;

		// first layer -> hidden layer
		for (i = hiddenIndex; i < outIndex; i++) {
			double sum = thresholds[i];
			for (j = 0; j < inputCount; j++) {
				sum += fire[j] * matrix[inx++];
			}
			fire[i] = threshold(sum);
		}

		// hidden layer -> output layer
		double result[] = new double[outputCount];

		for (i = outIndex; i < neuronCount; i++) {
			double sum = thresholds[i];
			for (j = hiddenIndex; j < outIndex; j++) {
				sum += fire[j] * matrix[inx++];
			}
			fire[i] = threshold(sum);
			result[i - outIndex] = fire[i];
		}
		return result;
	}

	/**
	 * Reset all weight and thresholds
	 */
	public void reset() {
		int i;
		for (i = 0; i < neuronCount; i++) {
			thresholds[i] = 0.5 - Math.random();
			thresholdDelta[i] = 0;
			accThresholdDelta[i] = 0;
		}

		for (i = 0; i < matrix.length; i++) {
			matrix[i] = 0.5 - Math.random();
			matrixDelta[i] = 0;
			accMatrixDelta[i] = 0;
		}
	}
}
