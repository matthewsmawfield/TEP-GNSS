// Solar System Visualization Class
// Using Three.js from CDN
class SolarSystemVisualization {
    constructor() {
        console.log('🚀 Initializing Solar System Visualization...');

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.planets = new Map();
        this.orbits = new Map();
        this.animationId = null;
        this.isPlaying = false;
        this.timeSpeed = 1;
        this.currentDate = new Date('2023-01-01');
        this.startDate = new Date('2023-01-01');
        this.endDate = new Date('2025-06-30');

        // Initialize components
        this.init();
        this.createSolarSystem();
        this.setupEventListeners();
        this.animate();

        console.log('✅ Solar System Visualization initialized successfully');

        // Planetary data (scaled for visualization)
        this.planetData = {
            sun: {
                radius: 10,
                color: 0xFFD700,
                emissive: 0x442211,
                position: [0, 0, 0]
            },
            mercury: {
                radius: 0.8,
                color: 0x8C7853,
                orbitRadius: 25,
                orbitSpeed: 4.15,
                orbitInclination: 7
            },
            venus: {
                radius: 1.2,
                color: 0xFFC649,
                orbitRadius: 35,
                orbitSpeed: 1.62,
                orbitInclination: 3.4
            },
            earth: {
                radius: 1.5,
                color: 0x6B93D6,
                orbitRadius: 50,
                orbitSpeed: 1.0,
                orbitInclination: 0
            },
            mars: {
                radius: 1.0,
                color: 0xCD5C5C,
                orbitRadius: 75,
                orbitSpeed: 0.53,
                orbitInclination: 1.8
            },
            jupiter: {
                radius: 6.0,
                color: 0xD8CA9D,
                orbitRadius: 150,
                orbitSpeed: 0.084,
                orbitInclination: 1.3
            },
            saturn: {
                radius: 5.0,
                color: 0xFAD5A5,
                orbitRadius: 220,
                orbitSpeed: 0.034,
                orbitInclination: 2.5
            }
        };

        this.init();
        this.createSolarSystem();
        this.setupEventListeners();
        this.animate();
    }

    init() {
        console.log('🔧 Setting up Three.js scene...');

        // Check if THREE is loaded
        if (typeof THREE === 'undefined') {
            throw new Error('THREE.js is not loaded. Check script loading order.');
        }

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000814);

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            2000
        );
        this.camera.position.set(100, 50, 100);

        // Get canvas element
        const canvas = document.getElementById('solarSystemCanvas');
        if (!canvas) {
            throw new Error('Canvas element not found. Check HTML structure.');
        }

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true
        });

        // Check if WebGL is supported
        if (!this.renderer) {
            throw new Error('WebGL not supported in this browser');
        }

        // Get the actual canvas dimensions from CSS
        const canvasWidth = canvas.clientWidth || window.innerWidth * 0.7;
        const canvasHeight = canvas.clientHeight || 600;
        this.renderer.setSize(canvasWidth, canvasHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Check if OrbitControls is loaded
        if (typeof OrbitControls === 'undefined') {
            throw new Error('OrbitControls is not loaded. Check script loading order.');
        }

        // Controls - try OrbitControls first, fallback to basic controls
        try {
            if (typeof OrbitControls !== 'undefined') {
                this.controls = new OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;
                this.controls.minDistance = 20;
                this.controls.maxDistance = 1000;
                console.log('✅ OrbitControls initialized');
            } else {
                throw new Error('OrbitControls not available');
            }
        } catch (error) {
            console.warn('⚠️ OrbitControls not available, using basic camera controls');
            this.setupBasicControls();
        }

        // Lighting
        this.setupLighting();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        console.log('✅ Three.js scene setup complete');
    }

    setupBasicControls() {
        // Basic mouse controls for camera
        this.mouse = { x: 0, y: 0, isDown: false };
        this.cameraDistance = 100;
        this.cameraAngle = { x: 0, y: 0 };

        const canvas = this.renderer.domElement;

        canvas.addEventListener('mousedown', (event) => {
            this.mouse.isDown = true;
            this.mouse.x = event.clientX;
            this.mouse.y = event.clientY;
        });

        canvas.addEventListener('mousemove', (event) => {
            if (this.mouse.isDown) {
                const deltaX = event.clientX - this.mouse.x;
                const deltaY = event.clientY - this.mouse.y;

                this.cameraAngle.y -= deltaX * 0.01;
                this.cameraAngle.x -= deltaY * 0.01;

                // Limit vertical rotation
                this.cameraAngle.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, this.cameraAngle.x));

                this.updateCameraPosition();
            }

            this.mouse.x = event.clientX;
            this.mouse.y = event.clientY;
        });

        canvas.addEventListener('mouseup', () => {
            this.mouse.isDown = false;
        });

        canvas.addEventListener('wheel', (event) => {
            event.preventDefault();
            this.cameraDistance += event.deltaY * 0.1;
            this.cameraDistance = Math.max(20, Math.min(500, this.cameraDistance));
            this.updateCameraPosition();
        });

        // Touch controls for mobile
        canvas.addEventListener('touchstart', (event) => {
            if (event.touches.length === 1) {
                this.mouse.isDown = true;
                this.mouse.x = event.touches[0].clientX;
                this.mouse.y = event.touches[0].clientY;
            }
        });

        canvas.addEventListener('touchmove', (event) => {
            if (this.mouse.isDown && event.touches.length === 1) {
                const deltaX = event.touches[0].clientX - this.mouse.x;
                const deltaY = event.touches[0].clientY - this.mouse.y;

                this.cameraAngle.y -= deltaX * 0.01;
                this.cameraAngle.x -= deltaY * 0.01;

                // Limit vertical rotation
                this.cameraAngle.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, this.cameraAngle.x));

                this.updateCameraPosition();
            }

            this.mouse.x = event.touches[0].clientX;
            this.mouse.y = event.touches[0].clientY;
        });

        canvas.addEventListener('touchend', () => {
            this.mouse.isDown = false;
        });

        this.updateCameraPosition();
    }

    updateCameraPosition() {
        this.camera.position.x = this.cameraDistance * Math.cos(this.cameraAngle.x) * Math.sin(this.cameraAngle.y);
        this.camera.position.y = this.cameraDistance * Math.sin(this.cameraAngle.x);
        this.camera.position.z = this.cameraDistance * Math.cos(this.cameraAngle.x) * Math.cos(this.cameraAngle.y);
        this.camera.lookAt(0, 0, 0);
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);

        // Sun light (point light at sun position)
        const sunLight = new THREE.PointLight(0xFFFFFF, 2, 1000);
        sunLight.castShadow = true;
        sunLight.shadow.mapSize.width = 2048;
        sunLight.shadow.mapSize.height = 2048;
        this.scene.add(sunLight);

        // Store reference for updating
        this.sunLight = sunLight;

        // Additional directional light for better illumination
        const directionalLight = new THREE.DirectionalLight(0xFFFFFF, 0.5);
        directionalLight.position.set(-100, 100, 50);
        this.scene.add(directionalLight);
    }

    createSolarSystem() {
        console.log('🌟 Creating solar system...');

        try {
            // Create Sun
            this.createSun();
            console.log('☀️ Sun created');

            // Create Planets
            Object.keys(this.planetData).forEach(planetName => {
                if (planetName !== 'sun') {
                    this.createPlanet(planetName);
                    this.createOrbit(planetName);
                    console.log(`🪐 ${planetName} created`);
                }
            });

            console.log('✅ Solar system creation complete');
        } catch (error) {
            console.error('❌ Error creating solar system:', error);
        }
    }

    createSun() {
        const sunData = this.planetData.sun;
        const sunGeometry = new THREE.SphereGeometry(sunData.radius, 32, 32);
        const sunMaterial = new THREE.MeshBasicMaterial({
            color: sunData.color,
            emissive: sunData.emissive,
            emissiveIntensity: 0.8
        });

        const sun = new THREE.Mesh(sunGeometry, sunMaterial);
        sun.position.set(0, 0, 0);
        this.scene.add(sun);
        this.planets.set('sun', sun);

        // Add sun glow effect
        this.addSunGlow();
    }

    addSunGlow() {
        const sunGlowGeometry = new THREE.SphereGeometry(12, 32, 32);
        const sunGlowMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                color: { value: new THREE.Color(0xFFD700) }
            },
            vertexShader: `
                varying vec3 vNormal;
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 color;
                varying vec3 vNormal;
                void main() {
                    float intensity = pow(0.8 - dot(vNormal, vec3(0, 0, 1.0)), 2.0);
                    gl_FragColor = vec4(color, intensity * 0.5);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending
        });

        const sunGlow = new THREE.Mesh(sunGlowGeometry, sunGlowMaterial);
        this.scene.add(sunGlow);
        this.sunGlow = sunGlow;
    }

    createPlanet(planetName) {
        const planetData = this.planetData[planetName];
        const geometry = new THREE.SphereGeometry(planetData.radius, 16, 16);
        const material = new THREE.MeshLambertMaterial({
            color: planetData.color,
            shininess: 30
        });

        const planet = new THREE.Mesh(geometry, material);
        planet.castShadow = true;
        planet.receiveShadow = true;

        // Store planet data
        planet.userData = {
            name: planetName,
            orbitRadius: planetData.orbitRadius,
            orbitSpeed: planetData.orbitSpeed,
            orbitInclination: planetData.orbitInclination,
            angle: Math.random() * Math.PI * 2 // Random starting position
        };

        this.planets.set(planetName, planet);
        this.scene.add(planet);
    }

    createOrbit(planetName) {
        const planetData = this.planetData[planetName];
        const orbitGeometry = new THREE.RingGeometry(
            planetData.orbitRadius - 0.1,
            planetData.orbitRadius + 0.1,
            64
        );

        const orbitMaterial = new THREE.MeshBasicMaterial({
            color: 0x444444,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.3
        });

        const orbit = new THREE.Mesh(orbitGeometry, orbitMaterial);
        orbit.rotation.x = -Math.PI / 2;
        orbit.rotation.z = planetData.orbitInclination * Math.PI / 180;

        this.orbits.set(planetName, orbit);
        this.scene.add(orbit);
    }

    updatePlanetaryPositions() {
        const daysSinceStart = Math.floor((this.currentDate - this.startDate) / (1000 * 60 * 60 * 24));

        this.planets.forEach((planet, planetName) => {
            if (planetName !== 'sun') {
                const planetData = this.planetData[planetName];
                const angle = planet.userData.angle + (daysSinceStart * planetData.orbitSpeed * 0.01 * this.timeSpeed);

                const x = Math.cos(angle) * planetData.orbitRadius;
                const z = Math.sin(angle) * planetData.orbitRadius;
                const y = Math.sin(angle) * planetData.orbitInclination * 0.5;

                planet.position.set(x, y, z);
                planet.rotation.y += 0.01 * this.timeSpeed;

                // Update orbit rotation
                if (this.orbits.has(planetName)) {
                    this.orbits.get(planetName).rotation.z = angle;
                }
            }
        });

        // Update sun light position
        if (this.sunLight) {
            this.sunLight.position.copy(this.planets.get('sun').position);
        }

        // Update sun glow animation
        if (this.sunGlow) {
            this.sunGlow.material.uniforms.time.value += 0.01;
        }
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());

        if (this.isPlaying) {
            this.updatePlanetaryPositions();

            // Advance time
            const msPerDay = 1000 * 60 * 60 * 24;
            const speedMultiplier = this.timeSpeed * 10; // Adjust for visible animation
            this.currentDate = new Date(this.currentDate.getTime() + (msPerDay / 60) * speedMultiplier);

            // Update date display
            this.updateDateDisplay();
        }

        // Update controls
        if (this.controls && this.controls.update) {
            this.controls.update(); // OrbitControls
        }
        // Basic controls are handled in event listeners

        this.renderer.render(this.scene, this.camera);
    }

    updateDateDisplay() {
        const dateElement = document.querySelector('.current-date');
        if (dateElement) {
            dateElement.textContent = this.currentDate.toISOString().split('T')[0];
        }
    }

    setupEventListeners() {
        // Play/Pause button
        const playPauseBtn = document.getElementById('playPauseBtn');
        if (playPauseBtn) {
            playPauseBtn.addEventListener('click', () => {
                this.isPlaying = !this.isPlaying;
                playPauseBtn.textContent = this.isPlaying ? '⏸️ Pause' : '▶️ Play';
            });
        }

        // Reset button
        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.currentDate = new Date('2023-01-01');
                this.isPlaying = false;
                playPauseBtn.textContent = '▶️ Play';
                this.updatePlanetaryPositions();
                this.updateDateDisplay();
            });
        }

        // Speed control
        const speedSlider = document.getElementById('speedSlider');
        const speedValue = document.getElementById('speedValue');
        if (speedSlider && speedValue) {
            speedSlider.addEventListener('input', (e) => {
                this.timeSpeed = parseFloat(e.target.value);
                speedValue.textContent = this.timeSpeed + 'x';
            });
        }
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();

        // Get updated canvas dimensions
        const canvas = this.renderer.domElement;
        const canvasWidth = canvas.clientWidth || window.innerWidth * 0.7;
        const canvasHeight = canvas.clientHeight || 600;
        this.renderer.setSize(canvasWidth, canvasHeight);
    }
}

// Show error message in loading overlay
function showError(message) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.innerHTML = `
            <div style="color: #ff6b6b; text-align: center; max-width: 300px;">
                <h3>⚠️ Loading Error</h3>
                <p>${message}</p>
                <button onclick="location.reload()" style="
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin-top: 10px;
                ">Refresh Page</button>
            </div>
        `;
    }
}

// Wait for scripts to load and then initialize
function initializeWhenReady(attempts = 0) {
    const maxAttempts = 100; // 10 seconds max
    console.log(`⏳ Checking if scripts are loaded... (attempt ${attempts + 1}/${maxAttempts})`);

    // Check if THREE.js is loaded
    if (typeof THREE === 'undefined') {
        if (attempts >= maxAttempts) {
            console.error('❌ THREE.js failed to load after maximum attempts');
            showError('THREE.js library failed to load. Please refresh the page.');
            return;
        }
        console.log('⏳ THREE.js not loaded yet, waiting...');
        setTimeout(() => initializeWhenReady(attempts + 1), 100);
        return;
    }

    // Check if OrbitControls is loaded
    if (typeof OrbitControls === 'undefined') {
        if (attempts >= maxAttempts) {
            console.error('❌ OrbitControls failed to load after maximum attempts');
            showError('OrbitControls library failed to load. Please refresh the page.');
            return;
        }
        console.log('⏳ OrbitControls not loaded yet, waiting...');
        setTimeout(() => initializeWhenReady(attempts + 1), 100);
        return;
    }

    console.log('✅ All scripts loaded! Initializing visualization...');

    // Hide loading overlay
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        setTimeout(() => {
            loadingOverlay.style.display = 'none';
        }, 500);
    }

    // Create the solar system visualization
    try {
        new SolarSystemVisualization();
        console.log('🎉 Solar System Visualization ready!');
    } catch (error) {
        console.error('💥 Failed to initialize visualization:', error);
        if (loadingOverlay) {
            loadingOverlay.innerHTML = `
                <div style="color: #ff6b6b; text-align: center;">
                    <h3>Initialization Error</h3>
                    <p>${error.message}</p>
                    <p>Please check the console for details.</p>
                </div>
            `;
        }
    }
}

// Initialize when dependencies are loaded (called from HTML after CDN scripts load)
function startSolarSystemVisualization() {
    console.log('🚀 Starting Solar System Visualization...');
    new SolarSystemVisualization();
}
