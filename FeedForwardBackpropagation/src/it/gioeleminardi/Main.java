package it.gioeleminardi;

import it.gioeleminardi.neural.Network;

import java.text.NumberFormat;

public class Main {

	public static void main(String[] args) {

		double xorInput[][] = {
				{0.0, 0.0},
				{0.0, 1.0},
				{1.0, 0.0},
				{1.0, 1.0}
		};

		double xorIdeal[][] = {
				{0.0},
				{1.0},
				{1.0},
				{0.0}
		};

		System.out.println("Learn: ");
		Network network = new Network(2, 3, 1, 0.7, 0.9);

		NumberFormat percentFormat = NumberFormat.getPercentInstance();
		percentFormat.setMinimumFractionDigits(4);

		for (int i = 0; i < 10000; i++) {
			for (int j = 0; j < xorInput.length; j++) {
				network.computeOutputs(xorInput[j]);
				network.calcError(xorIdeal[j]);
				network.learn();
			}
			System.out.printf("Trial #%d, Error: %s\n", i, percentFormat.format(network.getError(xorInput.length)));
		}

		System.out.println("End Training.");

		System.out.println("Results:");

		for (int i = 0; i < xorInput.length; i++) {
			for (int j = 0; j < xorInput[i].length; j++) {
				System.out.printf("%s: ", xorInput[i][j]);
			}
			double out[] = network.computeOutputs(xorInput[i]);
			System.out.printf("= %s\n", out[0]);
		}

	}

}
