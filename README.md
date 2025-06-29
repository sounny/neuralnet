# Interactive Neural Network Builder

A vanilla JavaScript implementation of an interactive neural network visualization tool. This project is designed as a hands‑on learning resource—build and experiment with neural networks, beginning with Martin's Image Recognition Machine (MIM).
Check out a live preview of the website [here](https://sounny.github.io/neuralnet/).

## Features

- **Interactive Network Visualization**: Visual representation of neural networks with neurons and weighted connections
- **Real-time Forward Pass**: See activations propagate through the network in real-time
- **Weight Editing**: Manually adjust weights and biases to see their effects
- **Multiple Activation Functions**: Support for Sigmoid, ReLU, Tanh, and Linear activation functions
- **Layer Management**: Add and remove hidden layers dynamically
- **Pattern Testing**: Test different input patterns (diagonal, horizontal, vertical, etc.)
- **Mathematical Breakdown**: View detailed calculations for each layer
- **Responsive Design**: Works on desktop and mobile devices

## Getting Started

### Prerequisites

- A modern web browser (Chrome, Firefox, Safari, Edge)
- No additional installation required!

### Running the Application

1. Clone or download this repository
2. Open `index.html` in your web browser
3. Start experimenting with the neural network!

### Usage

1. **Input Pattern**: Click the 4-pixel grid to create different patterns or use the preset buttons
2. **Network Controls**: 
   - Add hidden layers to make the network more complex
   - Remove layers to simplify
   - Reset to the original MIM configuration
3. **Activation Functions**: Switch between different activation functions to see their effects
4. **Weight Adjustment**: Modify weights in the weight matrices section
5. **Mathematical View**: Toggle the mathematical breakdown to see detailed calculations

## Learning Exercises

These short challenges encourage active exploration of the network. Try them out as you read the code:

1. **Predict before you click** – Before selecting a preset pattern, guess what the output values will be. Then run the pattern and compare your prediction.
2. **Add a hidden layer** – Insert a new hidden layer and observe how the activations and weights adjust.
3. **Experiment with activation functions** – Swap among Sigmoid, ReLU, Tanh and Linear to see how each changes the network’s behavior.
4. **Tweak the weights** – Manually modify a weight or bias and note how even small changes can alter the final output.

## About the MIM (Martin's Image Recognition Machine)

The default configuration implements Martin's Image Recognition Machine, designed to detect diagonal patterns in a 2x2 pixel grid:

This mini-network is intentionally simple so you can clearly see how information flows from inputs to outputs.

- **Input Layer**: 4 neurons representing the 2×2 image pixels.
- **Output Layer**: 2 neurons that light up when a diagonal pattern (\ or /) is detected.
- **Weights**: Values are arranged so that the first output fires for a left-to-right diagonal and the second fires for a right-to-left diagonal.

## Technical Details

- **Pure Vanilla JavaScript**: No frameworks or dependencies (except Lucide icons via CDN)
- **Real-time Calculations**: Forward pass is recalculated on every change
- **Responsive Design**: CSS Grid and Flexbox for modern layouts
- **SVG Visualizations**: Scalable connection lines between neurons

## File Structure

```
neuralnet/
├── index.html          # Main HTML file
├── style.css           # Styling and layout
├── script.js           # Neural network logic
└── README.md           # This file
```

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the [MIT License](LICENSE).
