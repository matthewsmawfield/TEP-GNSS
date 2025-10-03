# 3D Solar System Visualization

A professional, interactive 3D visualization of our solar system showing planetary positions during the TEP-GNSS analysis period (2023-2025). Built with Three.js and styled to match New York Times data visualization aesthetics.

## Features

- **Interactive 3D Solar System**: Full 3D models of the Sun, Earth, and major planets
- **Accurate Orbital Mechanics**: Realistic planetary positions calculated for the 2023-2025 timeframe
- **Time Animation**: Animated progression through the analysis period with playback controls
- **Professional Styling**: Clean, modern interface inspired by NYT data visualizations
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Controls**: Play/pause, speed adjustment, and date reset functionality

## Solar System Bodies Included

- **Sun**: Central star with glow effects
- **Mercury**: Innermost planet
- **Venus**: Second planet from the Sun
- **Earth**: Third planet (focus of TEP-GNSS analysis)
- **Mars**: Fourth planet
- **Jupiter**: Largest planet (gas giant)
- **Saturn**: Ringed planet

## Technical Implementation

### 3D Graphics
- **Three.js**: WebGL-based 3D rendering engine
- **OrbitControls**: Interactive camera controls
- **Realistic Lighting**: Point lights, ambient lighting, and shadow mapping
- **Custom Shaders**: Sun glow effects using fragment shaders

### Animation System
- **Time-based Animation**: Accurate planetary position calculations
- **Speed Control**: Variable playback speed (0.1x to 5x)
- **Date Range**: 2023-01-01 to 2025-06-30 (TEP-GNSS analysis period)
- **Real-time Updates**: Live date display and position updates

### Professional Design
- **Typography**: Inter font family for clean, readable text
- **Color Scheme**: Professional dark theme with subtle gradients
- **Layout**: Grid-based responsive layout
- **Interactive Elements**: Hover effects and smooth transitions

## Usage

### Running the Visualization

1. **Local Development**:
   ```bash
   cd scripts/exploratory/solar_system_3d_visualization
   python3 -m http.server 8348
   ```

2. **Open in Browser**:
   Navigate to `http://localhost:8348` in your web browser

### Controls

- **Mouse**: Click and drag to rotate the camera view
- **Scroll**: Zoom in and out
- **Play/Pause**: Start or stop the time animation
- **Reset**: Return to the starting date (2023-01-01)
- **Speed Control**: Adjust animation playback speed

### Interface Elements

- **Header**: Title, description, and animation controls
- **Date Display**: Shows current animation date
- **Planet Legend**: Color-coded reference for each celestial body
- **3D Canvas**: Interactive solar system visualization
- **Footer**: Project information and technical details

## File Structure

```
solar_system_3d_visualization/
├── index.html          # Main HTML structure
├── styles.css          # Professional CSS styling
├── solar-system.js     # Three.js visualization logic
├── package.json        # Project dependencies
└── README.md          # This documentation
```

## Dependencies

- **Three.js** (v0.158.0): 3D graphics library
- **OrbitControls**: Camera control extension
- **Modern Browser**: Chrome, Firefox, Safari, or Edge with WebGL support

## Performance Considerations

- **Optimized Rendering**: Efficient 3D rendering with appropriate level-of-detail
- **Memory Management**: Proper cleanup of Three.js resources
- **Responsive Design**: Adapts to different screen sizes
- **Progressive Loading**: Staggered asset loading for smooth user experience

## Future Enhancements

- **Additional Planets**: Uranus, Neptune, and dwarf planets
- **Satellite Visualization**: Earth's moon and other natural satellites
- **Data Integration**: Overlay TEP-GNSS analysis data on planetary positions
- **Advanced Controls**: Date picker, planet selection, and view presets
- **Export Features**: Screenshot and video export capabilities

## TEP-GNSS Context

This visualization serves as an interactive companion to the TEP-GNSS research project, providing spatial context for the temporal analysis of GNSS clock correlations across global networks during the 2023-2025 analysis period.

The animated solar system helps researchers visualize the geometric relationships between Earth, the Sun, and other planets that may influence the observed temporal equivalence principle effects in GNSS atomic clock data.
