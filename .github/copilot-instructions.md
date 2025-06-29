<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Neural Network Builder Project Instructions

This is a vanilla JavaScript neural network visualization project that implements an interactive neural network builder.

## Project Context

- **Technology Stack**: Pure HTML, CSS, and JavaScript (no frameworks)
- **Purpose**: Educational neural network visualization tool
- **Target**: Interactive learning and experimentation with neural networks
- **Base Implementation**: Martin's Image Recognition Machine (MIM) for diagonal pattern detection

## Code Style Guidelines

- Use modern ES6+ JavaScript features (classes, arrow functions, destructuring)
- Follow camelCase naming conventions
- Use semantic HTML5 elements
- Implement responsive CSS with Flexbox and Grid
- Maintain clean separation between HTML structure, CSS styling, and JavaScript logic

## Key Components

1. **NeuralNetworkBuilder Class**: Main application logic
2. **Network Visualization**: SVG-based visual representation
3. **Weight Matrices**: Interactive weight editing
4. **Activation Functions**: Multiple activation function support
5. **Forward Pass Calculation**: Real-time neural network computation

## Development Practices

- Prioritize readability and educational value
- Include comprehensive comments for mathematical operations
- Ensure cross-browser compatibility
- Maintain responsive design principles
- Use semantic variable and function names that reflect neural network concepts

## Performance Considerations

- Efficient DOM manipulation
- Minimal re-renders during weight updates
- Smooth animations for user interactions
- Optimized SVG rendering for network connections
