package it.gioeleminardi;

import it.gioeleminardi.neural.Network;

public class Main {

	public static void main(String[] args) {
		Network network = new Network(2,3,1,0.7,0.9);
		//System.out.printf(String.format("Network [input: %d, hidden: %d, output: %d, neurons: %d, weights: %d]"));
	}

}
