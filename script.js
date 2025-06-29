// ============================================================================
// Neural Network Builder - Vanilla JavaScript Implementation
// ============================================================================
// This class encapsulates all logic for constructing, visualizing,
// and interacting with a simple neural network (MIM) without any frameworks.
// Features:
//  - Dynamic layer management (add/remove hidden layers)
//  - Adjustable weights and biases via UI inputs
//  - Real-time forward pass calculation with multiple activation functions
//  - SVG-based network visualization and detailed math breakdown

class NeuralNetworkBuilder {
    /**
     * Initializes network state and UI on page load.
     */
    constructor() {
        // Network architecture state
        this.layers = [
            { type: 'input', neurons: 4, name: 'Input (4 pixels)' },
            { type: 'output', neurons: 2, name: 'Output (Diagonal Detection)' }
        ];
        
        // Weights between layers
        this.weights = {
            '0-1': [
                [0.5, -0.5], // Input neuron 0 to outputs
                [-0.5, 0.5], // Input neuron 1 to outputs
                [-0.5, 0.5], // Input neuron 2 to outputs
                [0.5, -0.5]  // Input neuron 3 to outputs
            ]
        };
        
        // Biases for each layer (except input)
        this.biases = {
            '1': [0, 0] // Output layer biases
        };
        
        // Current input pattern (4 pixels)
        this.inputPattern = [1, 0, 0, 1]; // Default diagonal
        
        // Computed activations for each layer
        this.activations = {};
        
        // UI state
        this.activationFunction = 'sigmoid';
        this.showCalculations = false;
        
        this.init();
    }
    
    /**
     * Entry point to set up event listeners and initial render.
     */
    init() {
        this.setupEventListeners();
        this.updatePixelGrid();
        this.calculateForwardPass();
        this.renderNetwork();
        this.renderWeightMatrices();
        this.renderResults();
        
        // Initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
    
    // Activation functions
    activationFunctions = {
        sigmoid: (x) => 1 / (1 + Math.exp(-x)),
        relu: (x) => Math.max(0, x),
        tanh: (x) => Math.tanh(x),
        linear: (x) => x
    };
    
    /**
     * Attaches event listeners to all UI controls (buttons, grid, selectors).
     */
    setupEventListeners() {
        // Control buttons
        document.getElementById('addLayerBtn').addEventListener('click', () => this.addHiddenLayer());
        document.getElementById('removeLayerBtn').addEventListener('click', () => this.removeHiddenLayer());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetToMIM());
        document.getElementById('showMathBtn').addEventListener('click', () => this.toggleMath());
        
        // Activation function selector
        document.getElementById('activationSelect').addEventListener('change', (e) => {
            this.activationFunction = e.target.value;
            this.calculateForwardPass();
            this.renderNetwork();
            this.renderResults();
            this.renderMathematicalBreakdown();
        });
        
        // Pixel grid
        document.getElementById('pixelGrid').addEventListener('click', (e) => {
            if (e.target.classList.contains('pixel')) {
                const index = parseInt(e.target.dataset.index);
                this.inputPattern[index] = 1 - this.inputPattern[index];
                this.updatePixelGrid();
                this.calculateForwardPass(true); // Animate on user interaction
                this.renderResults();
                this.renderMathematicalBreakdown();
            }
        });
        
        // Pattern buttons
        document.querySelectorAll('.pattern-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const pattern = e.target.dataset.pattern.split(',').map(Number);
                this.inputPattern = pattern;
                this.updatePixelGrid();
                this.calculateForwardPass(true); // Animate on pattern change
                this.renderResults();
                this.renderMathematicalBreakdown();
            });
        });
    }
    
    /**
     * Updates the pixel grid UI to reflect the current input pattern.
     */
    updatePixelGrid() {
        const pixels = document.querySelectorAll('.pixel');
        pixels.forEach((pixel, index) => {
            pixel.classList.toggle('active', this.inputPattern[index] === 1);
        });
    }
    
    /**
     * Performs a forward pass through all layers, computing activations.
     * @param {boolean} animate - Whether to show animations
     */
    calculateForwardPass(animate = false) {
        const newActivations = { '0': [...this.inputPattern] };
        
        for (let i = 1; i < this.layers.length; i++) {
            const prevLayerActivations = newActivations[`${i-1}`];
            const layerWeights = this.weights[`${i-1}-${i}`] || [];
            const layerBiases = this.biases[`${i}`] || [];
            
            const layerOutput = [];
            
            for (let j = 0; j < this.layers[i].neurons; j++) {
                let sum = layerBiases[j] || 0;
                
                for (let k = 0; k < prevLayerActivations.length; k++) {
                    const weight = layerWeights[k] ? layerWeights[k][j] : 0;
                    sum += prevLayerActivations[k] * weight;
                }
                
                const activated = this.activationFunctions[this.activationFunction](sum);
                layerOutput.push(activated);
            }
            
            newActivations[`${i}`] = layerOutput;
        }
        
        this.activations = newActivations;
        
        if (animate) {
            this.animateCalculation();
        }
    }

    /**
     * Triggers an animated forward pass visualization.
     */
    animateCalculation() {
        // Trigger network animation
        this.renderNetwork(true);
        
        // Update other components with a slight delay
        setTimeout(() => {
            this.renderResults();
            this.renderMathematicalBreakdown();
        }, 500);
    }
    
    /**
     * Adds a new hidden layer with random initial weights and zero biases.
     */
    addHiddenLayer() {
        const newLayers = [...this.layers];
        const hiddenLayer = { 
            type: 'hidden', 
            neurons: 3, 
            name: `Hidden ${newLayers.length - 1}` 
        };
        
        newLayers.splice(-1, 0, hiddenLayer);
        this.layers = newLayers;
        
        // Initialize weights and biases for new connections
        const newWeights = { ...this.weights };
        const newBiases = { ...this.biases };
        
        // Clear existing weights and rebuild
        Object.keys(newWeights).forEach(key => delete newWeights[key]);
        Object.keys(newBiases).forEach(key => delete newBiases[key]);
        
        // Rebuild weights and biases
        for (let i = 0; i < this.layers.length - 1; i++) {
            const fromLayer = this.layers[i];
            const toLayer = this.layers[i + 1];
            
            newWeights[`${i}-${i+1}`] = Array(fromLayer.neurons).fill().map(() => 
                Array(toLayer.neurons).fill().map(() => Math.random() * 2 - 1)
            );
            newBiases[`${i+1}`] = Array(toLayer.neurons).fill(0);
        }
        
        this.weights = newWeights;
        this.biases = newBiases;
        
        this.calculateForwardPass();
        this.renderNetwork();
        this.renderWeightMatrices();
        this.renderResults();
        this.renderMathematicalBreakdown();
        this.updateRemoveButtonState();
    }
    
    /**
     * Removes the last hidden layer and resets to MIM if only input and output remain.
     */
    removeHiddenLayer() {
        if (this.layers.length <= 2) return;
        
        const newLayers = this.layers.slice(0, -2).concat(this.layers.slice(-1));
        this.layers = newLayers;
        
        // Clean up weights and biases
        const newWeights = {};
        const newBiases = {};
        
        for (let i = 0; i < this.layers.length - 1; i++) {
            const fromLayer = this.layers[i];
            const toLayer = this.layers[i + 1];
            
            if (i === 0 && this.layers.length === 2) {
                // Direct input to output - use original MIM weights
                newWeights[`${i}-${i+1}`] = [
                    [0.5, -0.5],
                    [-0.5, 0.5],
                    [-0.5, 0.5],
                    [0.5, -0.5]
                ];
            } else {
                newWeights[`${i}-${i+1}`] = Array(fromLayer.neurons).fill().map(() => 
                    Array(toLayer.neurons).fill().map(() => Math.random() * 2 - 1)
                );
            }
            newBiases[`${i+1}`] = Array(toLayer.neurons).fill(0);
        }
        
        this.weights = newWeights;
        this.biases = newBiases;
        
        this.calculateForwardPass();
        this.renderNetwork();
        this.renderWeightMatrices();
        this.renderResults();
        this.renderMathematicalBreakdown();
        this.updateRemoveButtonState();
    }
    
    /**
     * Enables or disables the "Remove Layer" button based on layer count.
     */
    updateRemoveButtonState() {
        const removeBtn = document.getElementById('removeLayerBtn');
        removeBtn.disabled = this.layers.length <= 2;
    }
    
    /**
     * Resets network to original MIM configuration (2 layers, preset weights).
     */
    resetToMIM() {
        this.layers = [
            { type: 'input', neurons: 4, name: 'Input (4 pixels)' },
            { type: 'output', neurons: 2, name: 'Output (Diagonal Detection)' }
        ];
        this.weights = {
            '0-1': [
                [0.5, -0.5],
                [-0.5, 0.5],
                [-0.5, 0.5],
                [0.5, -0.5]
            ]
        };
        this.biases = { '1': [0, 0] };
        
        this.calculateForwardPass();
        this.renderNetwork();
        this.renderWeightMatrices();
        this.renderResults();
        this.renderMathematicalBreakdown();
        this.updateRemoveButtonState();
    }
    
    /**
     * Updates a specific weight value and triggers re-render.
     * @param {string} layerKey - Identifier for the layer connection (e.g., '0-1').
     * @param {number} fromNeuron - Index of the source neuron.
     * @param {number} toNeuron - Index of the target neuron.
     * @param {string|number} value - New weight value.
     */
    updateWeight(layerKey, fromNeuron, toNeuron, value) {
        if (!this.weights[layerKey]) {
            this.weights[layerKey] = [];
        }
        if (!this.weights[layerKey][fromNeuron]) {
            this.weights[layerKey][fromNeuron] = [];
        }
        this.weights[layerKey][fromNeuron][toNeuron] = parseFloat(value) || 0;
        
        this.calculateForwardPass(true); // Animate on weight change
        this.renderResults();
        this.renderMathematicalBreakdown();
    }
    
    /**
     * Renders the network structure: neurons and their connections (SVG lines).
     * @param {boolean} animate - Whether to show animations
     */
    renderNetwork(animate = false) {
        const container = document.getElementById('neuronsContainer');
        const svg = document.getElementById('networkSvg');
        
        // Clear existing content
        container.innerHTML = '';
        svg.innerHTML = '';
        
        const layerPositions = this.layers.map((layer, index) => ({
            x: index * 250 + 120, // Increased spacing
            y: 250, // Centered vertically
            neurons: layer.neurons
        }));
        
        // Store neuron positions for proper connections
        const neuronPositions = [];
        this.layers.forEach((layer, layerIndex) => {
            const position = layerPositions[layerIndex];
            const layerNeurons = [];
            
            for (let neuronIndex = 0; neuronIndex < layer.neurons; neuronIndex++) {
                const neuronY = position.y - (layer.neurons - 1) * 30 + neuronIndex * 60;
                layerNeurons.push({
                    x: position.x,
                    y: neuronY,
                    radius: 24
                });
            }
            neuronPositions.push(layerNeurons);
        });
        
        // Render connection lines with proper positioning
        this.layers.slice(0, -1).forEach((layer, layerIndex) => {
            const fromNeurons = neuronPositions[layerIndex];
            const toNeurons = neuronPositions[layerIndex + 1];
            const layerWeights = this.weights[`${layerIndex}-${layerIndex + 1}`] || [];
            
            layerWeights.forEach((neuronWeights, fromNeuron) => {
                neuronWeights.forEach((weight, toNeuron) => {
                    if (fromNeurons[fromNeuron] && toNeurons[toNeuron]) {
                        const fromPos = fromNeurons[fromNeuron];
                        const toPos = toNeurons[toNeuron];
                        
                        // Calculate connection points on neuron edges
                        const fromX = fromPos.x + fromPos.radius;
                        const fromY = fromPos.y;
                        const toX = toPos.x - toPos.radius;
                        const toY = toPos.y;
                        
                        const weightMagnitude = Math.abs(weight);
                        const isPositive = weight >= 0;
                        
                        // Connection line
                        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                        line.setAttribute('x1', fromX);
                        line.setAttribute('y1', fromY);
                        line.setAttribute('x2', toX);
                        line.setAttribute('y2', toY);
                        line.setAttribute('stroke', isPositive ? '#3b82f6' : '#ef4444');
                        line.setAttribute('stroke-width', Math.max(1, weightMagnitude * 4));
                        line.setAttribute('opacity', 0.4 + weightMagnitude * 0.6);
                        line.setAttribute('stroke-linecap', 'round');
                        
                        if (animate) {
                            line.classList.add('connection-animate');
                        }
                        
                        svg.appendChild(line);
                        
                        // Weight label
                        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                        text.setAttribute('x', (fromX + toX) / 2);
                        text.setAttribute('y', (fromY + toY) / 2 - 8);
                        text.setAttribute('text-anchor', 'middle');
                        text.setAttribute('font-size', '11');
                        text.setAttribute('font-weight', 'bold');
                        text.setAttribute('fill', '#374151');
                        text.setAttribute('stroke', 'white');
                        text.setAttribute('stroke-width', '2');
                        text.setAttribute('paint-order', 'stroke fill');
                        text.textContent = weight.toFixed(2);
                        svg.appendChild(text);
                    }
                });
            });
        });
        
        // Render neurons with improved positioning
        this.layers.forEach((layer, layerIndex) => {
            const position = layerPositions[layerIndex];
            
            const layerDiv = document.createElement('div');
            layerDiv.className = 'layer';
            layerDiv.style.left = `${position.x - 60}px`; // Adjusted for larger neurons
            layerDiv.style.top = `${position.y - (layer.neurons - 1) * 30 - 60}px`;
            
            const title = document.createElement('div');
            title.className = 'layer-title';
            title.textContent = layer.name;
            layerDiv.appendChild(title);
            
            const neuronsDiv = document.createElement('div');
            neuronsDiv.className = 'neurons';
            
            for (let neuronIndex = 0; neuronIndex < layer.neurons; neuronIndex++) {
                const activation = this.activations[`${layerIndex}`]?.[neuronIndex] || 0;
                const intensity = Math.abs(activation);
                const isPositive = activation >= 0;
                
                const neuronDiv = document.createElement('div');
                neuronDiv.className = `neuron ${isPositive ? 'positive' : 'negative'}`;
                neuronDiv.style.opacity = 0.4 + (intensity * 0.6);
                neuronDiv.style.transform = `scale(${0.9 + intensity * 0.3})`;
                neuronDiv.textContent = activation.toFixed(2);
                neuronDiv.title = `Layer ${layerIndex}, Neuron ${neuronIndex}: ${activation.toFixed(4)}`;
                neuronDiv.style.marginBottom = '12px'; // Increased spacing
                
                if (animate) {
                    // Add animation delay based on layer and neuron index
                    setTimeout(() => {
                        neuronDiv.classList.add('neuron-pulse');
                        setTimeout(() => neuronDiv.classList.remove('neuron-pulse'), 1000);
                    }, (layerIndex * 300) + (neuronIndex * 100));
                }
                
                neuronsDiv.appendChild(neuronDiv);
            }
            
            layerDiv.appendChild(neuronsDiv);
            container.appendChild(layerDiv);
        });
    }
    
    /**
     * Displays the output activations as colored bars with values.
     */
    renderResults() {
        const container = document.getElementById('outputResults');
        container.innerHTML = '';
        
        const outputActivations = this.activations[`${this.layers.length - 1}`] || [];
        
        outputActivations.forEach((output, index) => {
            const item = document.createElement('div');
            item.className = 'output-item';
            
            const label = document.createElement('span');
            label.className = 'output-label';
            label.textContent = `Output ${index}:`;
            
            const bar = document.createElement('div');
            bar.className = 'output-bar';
            
            const fill = document.createElement('div');
            fill.className = `output-fill ${output > 0 ? 'positive' : 'negative'}`;
            fill.style.width = `${Math.abs(output) * 100}%`;
            
            const value = document.createElement('span');
            value.className = 'output-value';
            value.textContent = output.toFixed(3);
            
            bar.appendChild(fill);
            item.appendChild(label);
            item.appendChild(bar);
            item.appendChild(value);
            container.appendChild(item);
        });
    }
    
    /**
     * Builds interactive weight matrix inputs for each layer connection.
     */
    renderWeightMatrices() {
        const container = document.getElementById('weightMatrices');
        container.innerHTML = '';
        
        this.layers.slice(0, -1).forEach((_, layerIndex) => {
            const fromLayer = layerIndex;
            const toLayer = layerIndex + 1;
            const layerKey = `${fromLayer}-${toLayer}`;
            const layerWeights = this.weights[layerKey] || [];
            
            const matrixDiv = document.createElement('div');
            matrixDiv.className = 'weight-matrix';
            
            const title = document.createElement('h4');
            title.textContent = `Weights: ${this.layers[fromLayer]?.name} → ${this.layers[toLayer]?.name}`;
            matrixDiv.appendChild(title);
            
            const grid = document.createElement('div');
            grid.className = 'weight-grid';
            
            layerWeights.forEach((neuronWeights, fromNeuron) => {
                const row = document.createElement('div');
                row.className = 'weight-row';
                
                const label = document.createElement('span');
                label.className = 'weight-label';
                label.textContent = `N${fromNeuron}:`;
                row.appendChild(label);
                
                neuronWeights.forEach((weight, toNeuron) => {
                    const input = document.createElement('input');
                    input.type = 'number';
                    input.step = '0.1';
                    input.value = weight.toFixed(2);
                    input.className = 'weight-input';
                    input.addEventListener('change', (e) => {
                        this.updateWeight(layerKey, fromNeuron, toNeuron, e.target.value);
                    });
                    row.appendChild(input);
                });
                
                grid.appendChild(row);
            });
            
            matrixDiv.appendChild(grid);
            container.appendChild(matrixDiv);
        });
    }
    
    /**
     * Toggles the display of detailed mathematical breakdown and updates UI.
     */
    toggleMath() {
        this.showCalculations = !this.showCalculations;
        const mathSection = document.getElementById('mathSection');
        const btn = document.getElementById('showMathBtn');
        
        if (this.showCalculations) {
            mathSection.style.display = 'block';
            btn.innerHTML = '<i data-lucide="book-open"></i> Hide Math';
            this.renderMathematicalBreakdown();
        } else {
            mathSection.style.display = 'none';
            btn.innerHTML = '<i data-lucide="book-open"></i> Show Math';
        }
        
        // Re-initialize icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
    
    /**
     * Populates the math section with calculation steps for each neuron.
     */
    renderMathematicalBreakdown() {
        if (!this.showCalculations) return;
        
        const container = document.getElementById('mathContent');
        container.innerHTML = '';
        
        this.layers.slice(1).forEach((layer, layerIndex) => {
            const actualLayerIndex = layerIndex + 1;
            const prevActivations = this.activations[`${actualLayerIndex - 1}`] || [];
            const currentActivations = this.activations[`${actualLayerIndex}`] || [];
            const layerWeights = this.weights[`${actualLayerIndex - 1}-${actualLayerIndex}`] || [];
            const layerBiases = this.biases[`${actualLayerIndex}`] || [];
            
            const layerDiv = document.createElement('div');
            layerDiv.className = 'math-layer';
            
            const title = document.createElement('h4');
            title.textContent = `Layer ${actualLayerIndex} (${layer.name}):`;
            layerDiv.appendChild(title);
            
            currentActivations.forEach((activation, neuronIndex) => {
                let calculation = `bias(${(layerBiases[neuronIndex] || 0).toFixed(2)})`;
                let sum = layerBiases[neuronIndex] || 0;
                
                prevActivations.forEach((prevActivation, prevIndex) => {
                    const weight = layerWeights[prevIndex]?.[neuronIndex] || 0;
                    calculation += ` + ${prevActivation.toFixed(2)} × ${weight.toFixed(2)}`;
                    sum += prevActivation * weight;
                });
                
                const neuronDiv = document.createElement('div');
                neuronDiv.className = 'math-neuron';
                neuronDiv.textContent = `Neuron ${neuronIndex}: ${calculation} = ${sum.toFixed(3)} → ${this.activationFunction}(${sum.toFixed(3)}) = ${activation.toFixed(3)}`;
                layerDiv.appendChild(neuronDiv);
            });
            
            container.appendChild(layerDiv);
        });
    }
}

// Initialize the neural network when the DOM is fully loaded
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    new NeuralNetworkBuilder();
});
