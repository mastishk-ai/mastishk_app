import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Maximize2, Minimize2, RotateCw, Zap, Eye, Settings } from 'lucide-react';

export function AdvancedThreeDViewer() {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showParticles, setShowParticles] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState([1.0]);
  const [layerSpacing, setLayerSpacing] = useState([2.0]);
  const [showConnections, setShowConnections] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  const threejsHTML = `
<!DOCTYPE html>
<html>
<head>
    <style>
        body { 
            margin: 0; 
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%); 
            font-family: 'Arial', sans-serif;
            overflow: hidden;
        }
        #container { 
            width: 100%; 
            height: ${isFullscreen ? '100vh' : '600px'}; 
            position: relative; 
            cursor: grab;
        }
        #container:active {
            cursor: grabbing;
        }
        #info { 
            position: absolute; 
            top: 20px; 
            left: 20px; 
            color: white; 
            background: rgba(0,0,0,0.8); 
            padding: 15px; 
            border-radius: 10px; 
            z-index: 100;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            font-size: 14px;
            line-height: 1.4;
        }
        #controls {
            position: absolute;
            bottom: 20px;
            right: 20px;
            color: white;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 10px;
            z-index: 100;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .metric { 
            display: flex; 
            justify-content: space-between; 
            margin: 5px 0;
        }
        .metric-value { 
            color: #4ECDC4; 
            font-weight: bold; 
        }
        .control-btn {
            background: rgba(78, 205, 196, 0.2);
            border: 1px solid #4ECDC4;
            color: #4ECDC4;
            padding: 8px 12px;
            border-radius: 5px;
            margin: 2px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .control-btn:hover {
            background: rgba(78, 205, 196, 0.4);
            transform: translateY(-2px);
        }
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        .status-active { background: #4ECDC4; }
        .status-training { background: #FF6B6B; }
        .status-idle { background: #96CEB4; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .particle-trail {
            position: absolute;
            width: 2px;
            height: 2px;
            background: #4ECDC4;
            border-radius: 50%;
            pointer-events: none;
            animation: trail 3s linear infinite;
        }
        
        @keyframes trail {
            0% { opacity: 1; transform: scale(1); }
            100% { opacity: 0; transform: scale(0.1); }
        }
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h4 style="margin: 0 0 10px 0; color: #4ECDC4;">🧠 Mastishk Transformer 3D</h4>
        <div class="metric">
            <span><span class="status-indicator status-active"></span>Status:</span>
            <span class="metric-value">Active</span>
        </div>
        <div class="metric">
            <span>Layers:</span>
            <span class="metric-value">32</span>
        </div>
        <div class="metric">
            <span>Parameters:</span>
            <span class="metric-value">7.2B</span>
        </div>
        <div class="metric">
            <span>Attention Heads:</span>
            <span class="metric-value">32</span>
        </div>
        <div class="metric">
            <span>Hidden Size:</span>
            <span class="metric-value">4096</span>
        </div>
        <hr style="border: 1px solid rgba(255,255,255,0.1); margin: 10px 0;">
        <div style="font-size: 12px; color: #96CEB4;">
            🖱️ Mouse: Rotate & Zoom<br>
            ⌨️ Keys: R (Reset) | P (Pause)<br>
            🎮 Scroll: Layer Spacing
        </div>
    </div>
    
    <div id="controls">
        <div style="margin-bottom: 10px; font-size: 12px; color: #96CEB4;">Real-time Controls</div>
        <button class="control-btn" onclick="resetCamera()">Reset View</button>
        <button class="control-btn" onclick="toggleAnimation()">Toggle Anim</button>
        <button class="control-btn" onclick="toggleParticles()">Particles</button>
        <button class="control-btn" onclick="cycleColorScheme()">Colors</button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        let scene, camera, renderer, controls;
        let transformerLayers = [];
        let particles = [];
        let animationSpeed = ${animationSpeed[0]};
        let layerSpacing = ${layerSpacing[0]};
        let isAnimating = true;
        let particlesEnabled = ${showParticles};
        let colorSchemeIndex = 0;
        
        const colorSchemes = [
            { primary: 0x4ECDC4, secondary: 0xFF6B6B, accent: 0x45B7D1, highlight: 0x96CEB4 },
            { primary: 0x667eea, secondary: 0x764ba2, accent: 0xf093fb, highlight: 0x4facfe },
            { primary: 0xfa709a, secondary: 0xfee140, accent: 0xa8edea, highlight: 0xd299c2 }
        ];
        
        function init() {
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0a);
            scene.fog = new THREE.Fog(0x0a0a0a, 10, 100);
            
            // Camera setup
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / ${isFullscreen ? 'window.innerHeight' : '600'}, 0.1, 1000);
            camera.position.set(0, 10, 20);
            
            // Renderer setup
            renderer = new THREE.WebGLRenderer({ 
                antialias: true, 
                alpha: true,
                shadowMap: true
            });
            renderer.setSize(window.innerWidth, ${isFullscreen ? 'window.innerHeight' : '600'});
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Enhanced lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(10, 10, 5);
            directionalLight.castShadow = true;
            directionalLight.shadow.mapSize.width = 2048;
            directionalLight.shadow.mapSize.height = 2048;
            scene.add(directionalLight);
            
            // Point lights for dramatic effect
            const pointLight1 = new THREE.PointLight(0x4ECDC4, 1, 50);
            pointLight1.position.set(-10, 5, 10);
            scene.add(pointLight1);
            
            const pointLight2 = new THREE.PointLight(0xFF6B6B, 1, 50);
            pointLight2.position.set(10, 5, -10);
            scene.add(pointLight2);
            
            // Create transformer architecture
            createTransformerLayers();
            
            // Initialize particles
            if (particlesEnabled) {
                initParticleSystem();
            }
            
            // Mouse controls
            setupControls();
            
            // Keyboard controls
            setupKeyboardControls();
            
            // Start animation
            animate();
        }
        
        function createTransformerLayers() {
            const currentScheme = colorSchemes[colorSchemeIndex];
            
            for (let i = 0; i < 32; i++) {
                const y = i * layerSpacing - 30;
                
                // Attention layer (torus with glow)
                const attentionGeometry = new THREE.TorusGeometry(1.5, 0.3, 16, 100);
                const attentionMaterial = new THREE.MeshPhongMaterial({ 
                    color: currentScheme.primary,
                    transparent: true, 
                    opacity: 0.8,
                    emissive: currentScheme.primary,
                    emissiveIntensity: 0.1
                });
                const attention = new THREE.Mesh(attentionGeometry, attentionMaterial);
                attention.position.set(-3, y, 0);
                attention.rotation.x = Math.PI / 2;
                attention.castShadow = true;
                attention.receiveShadow = true;
                scene.add(attention);
                
                // MLP layer (rounded box with glow)
                const mlpGeometry = new THREE.BoxGeometry(2, 0.4, 2);
                const mlpMaterial = new THREE.MeshPhongMaterial({ 
                    color: currentScheme.secondary,
                    transparent: true, 
                    opacity: 0.8,
                    emissive: currentScheme.secondary,
                    emissiveIntensity: 0.1
                });
                const mlp = new THREE.Mesh(mlpGeometry, mlpMaterial);
                mlp.position.set(3, y, 0);
                mlp.castShadow = true;
                mlp.receiveShadow = true;
                scene.add(mlp);
                
                // Connection lines
                if (${showConnections} && i < 31) {
                    const lineGeometry = new THREE.BufferGeometry().setFromPoints([
                        new THREE.Vector3(-3, y, 0),
                        new THREE.Vector3(-3, y + layerSpacing, 0)
                    ]);
                    const lineMaterial = new THREE.LineBasicMaterial({ 
                        color: currentScheme.accent,
                        transparent: true,
                        opacity: 0.6
                    });
                    const line = new THREE.Line(lineGeometry, lineMaterial);
                    scene.add(line);
                    
                    const lineGeometry2 = new THREE.BufferGeometry().setFromPoints([
                        new THREE.Vector3(3, y, 0),
                        new THREE.Vector3(3, y + layerSpacing, 0)
                    ]);
                    const line2 = new THREE.Line(lineGeometry2, lineMaterial);
                    scene.add(line2);
                }
                
                // Add to tracking array
                transformerLayers.push({ attention, mlp, layer: i });
            }
            
            // Input/Output layers
            const embedGeometry = new THREE.ConeGeometry(1, 2, 6);
            const embedMaterial = new THREE.MeshPhongMaterial({ 
                color: currentScheme.highlight,
                emissive: currentScheme.highlight,
                emissiveIntensity: 0.2
            });
            const inputEmbed = new THREE.Mesh(embedGeometry, embedMaterial);
            inputEmbed.position.set(0, -35, 0);
            scene.add(inputEmbed);
            
            const outputEmbed = new THREE.Mesh(embedGeometry, embedMaterial);
            outputEmbed.position.set(0, 35, 0);
            outputEmbed.rotation.z = Math.PI;
            scene.add(outputEmbed);
        }
        
        function initParticleSystem() {
            const particleGeometry = new THREE.BufferGeometry();
            const particleCount = 1000;
            const positions = new Float32Array(particleCount * 3);
            
            for (let i = 0; i < particleCount * 3; i++) {
                positions[i] = (Math.random() - 0.5) * 100;
            }
            
            particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            const particleMaterial = new THREE.PointsMaterial({
                color: 0x4ECDC4,
                size: 0.1,
                transparent: true,
                opacity: 0.6
            });
            
            const particleSystem = new THREE.Points(particleGeometry, particleMaterial);
            scene.add(particleSystem);
            particles.push(particleSystem);
        }
        
        function setupControls() {
            let mouseX = 0, mouseY = 0, isMouseDown = false;
            let targetRotationX = 0, targetRotationY = 0;
            
            document.addEventListener('mousemove', (event) => {
                if (!isMouseDown) return;
                
                const deltaX = event.clientX - mouseX;
                const deltaY = event.clientY - mouseY;
                
                targetRotationY += deltaX * 0.01;
                targetRotationX += deltaY * 0.01;
                
                mouseX = event.clientX;
                mouseY = event.clientY;
            });
            
            document.addEventListener('mousedown', (event) => {
                isMouseDown = true;
                mouseX = event.clientX;
                mouseY = event.clientY;
            });
            
            document.addEventListener('mouseup', () => {
                isMouseDown = false;
            });
            
            document.addEventListener('wheel', (event) => {
                camera.position.z += event.deltaY * 0.01;
                camera.position.z = Math.max(5, Math.min(50, camera.position.z));
            });
            
            // Smooth rotation
            function updateRotation() {
                scene.rotation.y += (targetRotationY - scene.rotation.y) * 0.1;
                scene.rotation.x += (targetRotationX - scene.rotation.x) * 0.1;
                requestAnimationFrame(updateRotation);
            }
            updateRotation();
        }
        
        function setupKeyboardControls() {
            document.addEventListener('keydown', (event) => {
                switch(event.key.toLowerCase()) {
                    case 'r':
                        resetCamera();
                        break;
                    case 'p':
                        toggleAnimation();
                        break;
                    case 'c':
                        cycleColorScheme();
                        break;
                }
            });
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            const time = Date.now() * 0.001 * animationSpeed;
            
            if (isAnimating) {
                // Animate transformer layers
                transformerLayers.forEach((layer, index) => {
                    const offset = index * 0.1;
                    layer.attention.rotation.z = time + offset;
                    layer.mlp.rotation.y = Math.sin(time + offset) * 0.3;
                    
                    // Pulsing effect
                    const pulse = Math.sin(time * 2 + offset) * 0.1 + 1;
                    layer.attention.scale.setScalar(pulse);
                    layer.mlp.scale.setScalar(pulse);
                });
                
                // Animate particles
                particles.forEach(particleSystem => {
                    particleSystem.rotation.y = time * 0.1;
                    particleSystem.rotation.x = time * 0.05;
                });
            }
            
            renderer.render(scene, camera);
        }
        
        // Control functions
        function resetCamera() {
            camera.position.set(0, 10, 20);
            scene.rotation.set(0, 0, 0);
        }
        
        function toggleAnimation() {
            isAnimating = !isAnimating;
            document.querySelector('.control-btn:nth-child(2)').textContent = 
                isAnimating ? 'Pause Anim' : 'Start Anim';
        }
        
        function toggleParticles() {
            particlesEnabled = !particlesEnabled;
            particles.forEach(p => p.visible = particlesEnabled);
        }
        
        function cycleColorScheme() {
            colorSchemeIndex = (colorSchemeIndex + 1) % colorSchemes.length;
            const currentScheme = colorSchemes[colorSchemeIndex];
            
            // Update layer colors
            transformerLayers.forEach(layer => {
                layer.attention.material.color.setHex(currentScheme.primary);
                layer.attention.material.emissive.setHex(currentScheme.primary);
                layer.mlp.material.color.setHex(currentScheme.secondary);
                layer.mlp.material.emissive.setHex(currentScheme.secondary);
            });
        }
        
        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / ${isFullscreen ? 'window.innerHeight' : '600'};
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, ${isFullscreen ? 'window.innerHeight' : '600'});
        });
        
        // Initialize
        init();
    </script>
</body>
</html>`;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Advanced 3D Neural Network
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                Live Rendering
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsFullscreen(!isFullscreen)}
              >
                {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
              </Button>
            </div>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Interactive transformer layers with real-time animation and particle systems
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Controls Panel */}
            <div className="space-y-4">
              <div className="space-y-3">
                <Label className="flex items-center gap-2">
                  <Settings className="w-4 h-4" />
                  Animation Controls
                </Label>
                
                <div className="space-y-2">
                  <Label className="text-sm">Animation Speed: {animationSpeed[0].toFixed(1)}x</Label>
                  <Slider
                    value={animationSpeed}
                    onValueChange={setAnimationSpeed}
                    max={3}
                    min={0.1}
                    step={0.1}
                    className="w-full"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label className="text-sm">Layer Spacing: {layerSpacing[0].toFixed(1)}</Label>
                  <Slider
                    value={layerSpacing}
                    onValueChange={setLayerSpacing}
                    max={4}
                    min={1}
                    step={0.1}
                    className="w-full"
                  />
                </div>
              </div>

              <div className="space-y-3">
                <Label className="flex items-center gap-2">
                  <Eye className="w-4 h-4" />
                  Visual Effects
                </Label>
                
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="particles" 
                    checked={showParticles}
                    onCheckedChange={(checked) => setShowParticles(checked === true)}
                  />
                  <Label htmlFor="particles" className="text-sm">Particle System</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="connections" 
                    checked={showConnections}
                    onCheckedChange={(checked) => setShowConnections(checked === true)}
                  />
                  <Label htmlFor="connections" className="text-sm">Layer Connections</Label>
                </div>
              </div>

              <div className="space-y-2 pt-4 border-t">
                <Label>Advanced Features</Label>
                <div className="grid grid-cols-1 gap-2">
                  <Button variant="outline" size="sm" className="w-full">
                    <RotateCw className="w-4 h-4 mr-2" />
                    Auto-Rotate
                  </Button>
                  <Button variant="outline" size="sm" className="w-full">
                    <Eye className="w-4 h-4 mr-2" />
                    X-Ray Mode
                  </Button>
                  <Button variant="outline" size="sm" className="w-full">
                    <Zap className="w-4 h-4 mr-2" />
                    Data Flow
                  </Button>
                </div>
              </div>

              <div className="space-y-2 pt-4 border-t">
                <Label>Architecture Info</Label>
                <div className="text-sm space-y-1">
                  <div className="flex justify-between">
                    <span>Layers:</span>
                    <span className="font-mono">32</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Parameters:</span>
                    <span className="font-mono">7.2B</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Heads:</span>
                    <span className="font-mono">32</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Hidden:</span>
                    <span className="font-mono">4096</span>
                  </div>
                </div>
              </div>
            </div>

            {/* 3D Viewer */}
            <div className="lg:col-span-3">
              <div 
                ref={containerRef}
                className={`bg-card rounded-lg overflow-hidden border ${
                  isFullscreen ? 'fixed inset-0 z-50 bg-black' : ''
                }`}
                style={{ height: isFullscreen ? '100vh' : '600px' }}
              >
                <div
                  dangerouslySetInnerHTML={{ __html: threejsHTML }}
                  className="w-full h-full"
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}