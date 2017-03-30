package it.gioeleminardi.neural;

/**
 * Created by gioelem3 on 29/03/17.
 * See {@url https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/}
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
		final int hiddenIndex = inputCount;
		final int outIndex = inputCount + hiddenCount;

		for (i = 0; i < inputCount; i++) { //i scorre sull'input layer
			fire[i] = input[i]; //all'inizio fire, tra 0 e inputCount contiene gli input (primo giro della neural net)
		}

		int inx = 0; //indice globale per la matrix dei pesi

		// first layer -> hidden layer
		for (i = hiddenIndex; i < outIndex; i++) { //i scorre sull'hidden layer
			double sum = thresholds[i]; //prendo il precedente threshold
			for (j = 0; j < inputCount; j++) { //j scorre sull'input layer
				sum += fire[j] * matrix[inx++]; //moltiplico il j-esimo valore di input * il peso dell'arco tra il neurone hidden i e l'input j
			}
			fire[i] = threshold(sum); //threshold torna tra 0 e 1
		}

		// hidden layer -> output layer
		double result[] = new double[outputCount];

		for (i = outIndex; i < neuronCount; i++) { //i scorre sull'output layer
			double sum = thresholds[i];
			for (j = hiddenIndex; j < outIndex; j++) { //j scorre sull'hidden layer
				sum += fire[j] * matrix[inx++];
			}
			fire[i] = threshold(sum);
			result[i - outIndex] = fire[i];
		}
		return result;
	}

	/**
	 * Calculate the error for the recognition just done
	 *
	 * @param ideal What the output neurons should have yielded
	 */
	public void calcError(double ideal[]) {
		int i, j;
		final int hiddenIndex = inputCount;
		final int outputIndex = inputCount + hiddenCount;

		// clear hidden layer errors
		for (i = inputCount; i < neuronCount; i++) { //i scorre su hidden layer + output layer
			error[i] = 0;
		}

		// layer errors and deltas for the output layer
		for (i = outputIndex; i < neuronCount; i++) { //i scorre sull'output layer
			error[i] = ideal[i - outputIndex] - fire[i]; //calcolo l'errore per ogni neurone di output
			globalError += error[i] * error[i]; //calcolo l'errore globale
			errorDelta[i] = error[i] * fire[i] * (1 - fire[i]); //calcolo il delta di ogni errore
		}

		// hidden layer errors
		int winx = inputCount * hiddenCount; //indice per la matrice dei pesi (matrix). winx parte dai pesi tra hidden layer e output layer

		for (i = outputIndex; i < neuronCount; i++) { //i scorre sull'output layer
			for (j = hiddenIndex; j < outputIndex; j++) { //j scorre sull'hidden layer
				accMatrixDelta[winx] += errorDelta[i] * fire[j];
				error[j] += matrix[winx] * errorDelta[i];
				winx++;
			}
		}

		// hidden layer deltas
		for (i = hiddenIndex; i < outputIndex; i++) { //i scorre sull'hidden layer
			errorDelta[i] = error[i] * fire[i] * (1 - fire[i]);
		}

		// input layer errors
		winx = 0; //matrix con winx=0 parte dai pesi tra input layer e hidden layer
		for (i = hiddenIndex; i < outputIndex; i++) { //i scorre l'hidden layer
			for (j = 0; j < hiddenIndex; j++) {//j scorre l'input layer
				accMatrixDelta[winx] += errorDelta[i] + fire[j];
				error[j] += matrix[winx] * errorDelta[i];
				winx++;
			}
			accThresholdDelta[i] += errorDelta[i];
		}

	}

	/**
	 * Modify the weight matrix and thresholds bases on the last call to calcError
	 */
	public void learn() {
		int i;

		// process the matrix
		for (i = 0; i < matrix.length; i++) {
			matrixDelta[i] = (learnRate * accMatrixDelta[i]) + (momentum * matrixDelta[i]);
			matrix[i] += matrixDelta[i];
			accMatrixDelta[i] = 0;
		}

		// process the thresholds
		for (i = inputCount; i < neuronCount; i++) {
			thresholdDelta[i] = (learnRate * accThresholdDelta[i]) + (momentum * matrixDelta[i]);
			thresholds[i] += thresholdDelta[i];
			accThresholdDelta[i] = 0;
		}
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
