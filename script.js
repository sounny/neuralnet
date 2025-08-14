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

        // Educational labels and descriptions
        this.outputLabels = ['Diagonal \\', 'Diagonal /'];
        this.activationDescriptions = {
            sigmoid: 'Sigmoid squashes values between 0 and 1, useful for probabilities.',
            relu: 'ReLU outputs zero for negatives and the input for positives.',
            tanh: 'Tanh outputs values between -1 and 1, centered at zero.',
            linear: 'Linear passes values through unchanged.'
        };

        this.isDarkMode = localStorage.getItem('theme') === 'dark';

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
        this.updateActivationInfo();
        this.initTheme();

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
        document.getElementById('themeToggle').addEventListener('click', () => this.toggleTheme());

        // Activation function selector
        document.getElementById('activationSelect').addEventListener('change', (e) => {
            this.activationFunction = e.target.value;
            this.calculateForwardPass();
            this.renderNetwork();
            this.renderResults();
            this.renderMathematicalBreakdown();
            this.updateActivationInfo();
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
        const neuronsContainer = document.getElementById('neuronsContainer');
        const svg = document.getElementById('networkSvg');
        
        // Clear existing content
        neuronsContainer.innerHTML = '';
        svg.innerHTML = '';

        const neuronRadius = 24; // Defined from CSS .neuron width/height / 2
        const neuronDiameter = neuronRadius * 2;
        const verticalNeuronMargin = 12; // Defined from CSS .neuron style.marginBottom
        const layerTitleHeight = 30; // Approximate height for layer title + margin
        const layerWidth = 128; // Defined from CSS .layer-title width

        const svgRect = svg.getBoundingClientRect();
        const containerRect = neuronsContainer.getBoundingClientRect();

        const layerXSpacing = (svgRect.width - layerWidth) / Math.max(1, this.layers.length -1);

        const layerPositions = this.layers.map((layer, index) => {
            // Distribute layers horizontally across the available SVG width
            let x = layerWidth / 2; // Initial offset for the first layer
            if (this.layers.length > 1) {
                x = (index * layerXSpacing) + (layerWidth / 2);
            }
            // Ensure first layer is not too far left and last layer not too far right
            x = Math.max(layerWidth/2 + neuronRadius, Math.min(x, svgRect.width - layerWidth/2 - neuronRadius));


            return {
                x: x,
                // y: svgRect.height / 2, // Vertically center the layer block
                y: containerRect.height / 2, // Vertically center the layer block in its container
                neurons: layer.neurons,
                name: layer.name
            };
        });
        
        // Render neurons for each layer
        this.layers.forEach((layer, layerIndex) => {
            const layerConfig = layerPositions[layerIndex];
            const numNeurons = layer.neurons;
            const totalLayerHeight = (numNeurons * neuronDiameter) + ((numNeurons - 1) * verticalNeuronMargin);
            
            const layerDiv = document.createElement('div');
            layerDiv.className = 'layer';
            // Position the layer div. Its top-left is (0,0) relative to neuronsContainer
            // The neurons inside will be positioned relative to this layerDiv.
            // The calculated x for layerConfig is the center of the layer.
            // The calculated y for layerConfig is the center of the neurons block.
            layerDiv.style.position = 'absolute';
            layerDiv.style.left = `${layerConfig.x - layerWidth / 2}px`;
            layerDiv.style.top = `${layerConfig.y - totalLayerHeight / 2 - layerTitleHeight}px`; // Adjust for title
            layerDiv.style.width = `${layerWidth}px`; // Set fixed width for alignment

            const title = document.createElement('div');
            title.className = 'layer-title';
            title.textContent = layer.name;
            layerDiv.appendChild(title);
            
            const neuronsDiv = document.createElement('div');
            neuronsDiv.className = 'neurons'; // This will be a flex container centered by its parent
            
            for (let neuronIndex = 0; neuronIndex < numNeurons; neuronIndex++) {
                const activation = this.activations[`${layerIndex}`]?.[neuronIndex] || 0;
                const intensity = Math.abs(activation);
                const isPositive = activation >= 0;
                
                const neuronDiv = document.createElement('div');
                neuronDiv.className = `neuron ${isPositive ? 'positive' : 'negative'}`;
                // Neuron styles (width, height, margin) are from CSS
                neuronDiv.style.opacity = 0.4 + (intensity * 0.6);
                neuronDiv.style.transform = `scale(${0.9 + intensity * 0.3})`;
                neuronDiv.textContent = activation.toFixed(2);
                neuronDiv.title = `Layer ${layerIndex}, Neuron ${neuronIndex}: ${activation.toFixed(4)}`;
                
                if (animate) {
                    // Pulse input layer neurons early, subsequent layers after signal arrival time
                    const pulseDelay = (layerIndex === 0 ? 0 : ((layerIndex - 1) * 300) + 750) + (neuronIndex * 100);
                    setTimeout(() => {
                        neuronDiv.classList.add('neuron-pulse');
                        setTimeout(() => neuronDiv.classList.remove('neuron-pulse'), 1000); // Duration of pulse animation
                    }, pulseDelay);
                }
                
                neuronsDiv.appendChild(neuronDiv);
            }
            layerDiv.appendChild(neuronsDiv);
            neuronsContainer.appendChild(layerDiv);
        });

        // Calculate global positions for all neurons after rendering
        const neuronGlobalPositions = [];
        const updatedSvgRect = svg.getBoundingClientRect();
        neuronsContainer.querySelectorAll('.layer').forEach((layerEl, layerIndex) => {
            const neuronPositions = [];
            layerEl.querySelectorAll('.neuron').forEach((neuronEl) => {
                const rect = neuronEl.getBoundingClientRect();
                neuronPositions.push({
                    x: rect.left - updatedSvgRect.left + rect.width / 2,
                    y: rect.top - updatedSvgRect.top + rect.height / 2,
                    radius: rect.width / 2
                });
            });
            neuronGlobalPositions[layerIndex] = neuronPositions;
        });

        // Render connection lines using global neuron positions
        this.layers.slice(0, -1).forEach((_, layerIndex) => {
            const fromLayerNeurons = neuronGlobalPositions[layerIndex];
            const toLayerNeurons = neuronGlobalPositions[layerIndex + 1];
            const layerWeights = this.weights[`${layerIndex}-${layerIndex + 1}`] || [];

            if (!fromLayerNeurons || !toLayerNeurons) return;

            layerWeights.forEach((neuronWeights, fromNeuronIndex) => {
                neuronWeights.forEach((weight, toNeuronIndex) => {
                    const fromPos = fromLayerNeurons[fromNeuronIndex];
                    const toPos = toLayerNeurons[toNeuronIndex];

                    if (fromPos && toPos) {
                        // Line connects center to center of neurons
                        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                        line.setAttribute('x1', fromPos.x.toString());
                        line.setAttribute('y1', fromPos.y.toString());
                        line.setAttribute('x2', toPos.x.toString());
                        line.setAttribute('y2', toPos.y.toString());

                        const weightMagnitude = Math.abs(weight);
                        const isPositive = weight >= 0;
                        line.setAttribute('stroke', isPositive ? '#3b82f6' : '#ef4444');
                        line.setAttribute('stroke-width', Math.max(1, weightMagnitude * 4).toString());
                        line.setAttribute('opacity', (0.4 + weightMagnitude * 0.6).toString());
                        line.setAttribute('stroke-linecap', 'round');

                        if (animate) {
                            // Remove old class-based animation for the line itself
                            // line.classList.add('connection-animate');

                            // Add signal dot animation
                            const fromActivation = this.activations[`${layerIndex}`]?.[fromNeuronIndex] || 0;
                            if (Math.abs(fromActivation) > 0.01) { // Only animate if activation is significant
                                const signalDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                                signalDot.setAttribute('r', (3 + Math.abs(fromActivation) * 4).toString()); // Size based on activation
                                signalDot.setAttribute('fill', isPositive ? '#3b82f6' : '#ef4444'); // Color based on weight sign
                                signalDot.style.opacity = (0.6 + Math.abs(fromActivation) * 0.4).toString();

                                const motion = document.createElementNS('http://www.w3.org/2000/svg', 'animateMotion');
                                motion.setAttribute('path', `M${fromPos.x},${fromPos.y} L${toPos.x},${toPos.y}`);
                                motion.setAttribute('dur', '0.75s'); // Duration of travel
                                motion.setAttribute('begin', `${(layerIndex * 0.3)}s`); // Stagger start based on layer
                                motion.setAttribute('fill', 'freeze'); // Keep dot at end position
                                motion.setAttribute('calcMode', 'spline');
                                motion.setAttribute('keyTimes', '0;1');
                                motion.setAttribute('keySplines', '0.4 0 0.2 1');


                                signalDot.appendChild(motion);
                                svg.appendChild(signalDot);

                                // Optional: Fade out the dot after animation
                                setTimeout(() => {
                                   if (signalDot.parentNode) { // Check if still in DOM
                                       signalDot.style.transition = 'opacity 0.5s';
                                       signalDot.style.opacity = '0';
                                       setTimeout(() => {
                                           if (signalDot.parentNode) signalDot.remove();
                                       }, 500);
                                   }
                                }, (layerIndex * 300) + 750 + 200); // Start fadeout after arrival + delay
                            }
                        }
                        svg.appendChild(line);

                        // Weight label
                        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                        text.setAttribute('x', ((fromPos.x + toPos.x) / 2).toString());
                        text.setAttribute('y', ((fromPos.y + toPos.y) / 2 - 8).toString()); // Offset slightly above line
                        text.setAttribute('text-anchor', 'middle');
                        text.setAttribute('font-size', '11px');
                        text.setAttribute('font-weight', 'bold');
                        text.setAttribute('fill', '#374151');
                        text.setAttribute('stroke', 'white');
                        text.setAttribute('stroke-width', '2px');
                        text.setAttribute('paint-order', 'stroke fill');
                        text.textContent = weight.toFixed(2);
                        svg.appendChild(text);
                    }
                });
            });
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
            label.textContent = `${this.outputLabels[index] || 'Output ' + index}:`;

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

        const predictionEl = document.getElementById('predictionText');
        if (predictionEl) {
            if (outputActivations.length) {
                const maxVal = Math.max(...outputActivations);
                const maxIndex = outputActivations.indexOf(maxVal);
                if (maxVal >= 0.5) {
                    predictionEl.textContent = `${this.outputLabels[maxIndex] || 'Pattern'} detected (confidence ${maxVal.toFixed(2)})`;
                } else {
                    predictionEl.textContent = 'No strong pattern detected';
                }
            } else {
                predictionEl.textContent = '';
            }
        }
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

    updateActivationInfo() {
        const info = document.getElementById('activationInfo');
        if (info) {
            info.textContent = this.activationDescriptions[this.activationFunction] || '';
        }
    }

    initTheme() {
        document.body.classList.toggle('dark', this.isDarkMode);
        this.updateThemeButton();
    }

    toggleTheme() {
        this.isDarkMode = !this.isDarkMode;
        document.body.classList.toggle('dark', this.isDarkMode);
        localStorage.setItem('theme', this.isDarkMode ? 'dark' : 'light');
        this.updateThemeButton();
    }

    updateThemeButton() {
        const btn = document.getElementById('themeToggle');
        if (btn) {
            btn.innerHTML = this.isDarkMode
                ? '<i data-lucide="sun"></i> Light Mode'
                : '<i data-lucide="moon"></i> Dark Mode';
            if (typeof lucide !== 'undefined') {
                lucide.createIcons();
            }
        }
    }
}

// Link glossary keywords in page text
// ---------------------------------------------------------------------------
const glossaryLinks = {
    'neural network': 'glossary-neural-network',
    neuron: 'glossary-neuron',
    layers: 'glossary-layers',
    weight: 'glossary-weight',
    bias: 'glossary-bias',
    'activation function': 'glossary-activation-function',
    'forward pass': 'glossary-forward-pass',
    mim: 'glossary-mim'
};

const linkGlossaryTerms = () => {
    const textNodes = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
        const parent = node.parentElement;
        if (
            parent &&
            !parent.closest('.glossary-section') &&
            !['SCRIPT', 'STYLE', 'A'].includes(parent.tagName)
        ) {
            textNodes.push(node);
        }
    }

    textNodes.forEach((textNode) => {
        let content = textNode.textContent;
        Object.entries(glossaryLinks).forEach(([term, id]) => {
            const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`\\b${escaped}\\b`, 'gi');
            content = content.replace(regex, (match) => `<a href="#${id}" class="keyword"><em><strong>${match}</strong></em></a>`);
        });
        if (content !== textNode.textContent) {
            const span = document.createElement('span');
            span.innerHTML = content;
            textNode.parentElement.replaceChild(span, textNode);
        }
    });
};

// Initialize the neural network when the DOM is fully loaded
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    new NeuralNetworkBuilder();
    linkGlossaryTerms();
});
