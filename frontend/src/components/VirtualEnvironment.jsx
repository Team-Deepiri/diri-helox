import React, { Suspense, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sky, Environment, Stars, Cloud, Text } from '@react-three/drei';
import { useVirtualEnvironment } from '../hooks/useVirtualEnvironment';
import './VirtualEnvironment.css';

/**
 * Particle System Component
 */
function ParticleSystem({ type, count = 1000 }) {
  const particles = useRef();

  useFrame((state) => {
    if (particles.current) {
      particles.current.rotation.y += 0.001;
    }
  });

  const particleConfigs = {
    speed_lines: {
      color: 0x00ffff,
      speed: 0.05,
      size: 0.1
    },
    glowing_puzzles: {
      color: 0xffd700,
      speed: 0.02,
      size: 0.2
    },
    color_splashes: {
      color: 0xff6b9d,
      speed: 0.03,
      size: 0.15
    },
    matrix_rain: {
      color: 0x00ff00,
      speed: 0.1,
      size: 0.05
    },
    leaves: {
      color: 0x90ee90,
      speed: 0.02,
      size: 0.1
    },
    stars: {
      color: 0xffffff,
      speed: 0.01,
      size: 0.05
    },
    magic_sparks: {
      color: 0xff6b35,
      speed: 0.04,
      size: 0.1
    },
    bubbles: {
      color: 0x00ccff,
      speed: 0.02,
      size: 0.15
    },
    neon_particles: {
      color: 0xff00ff,
      speed: 0.05,
      size: 0.1
    },
    cherry_blossoms: {
      color: 0xffb6c1,
      speed: 0.02,
      size: 0.1
    }
  };

  const config = particleConfigs[type] || particleConfigs.stars;

  return (
    <points ref={particles}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={new Float32Array(count * 3).map(() => (Math.random() - 0.5) * 20)}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={config.size} color={config.color} />
    </points>
  );
}

/**
 * Weather System Component
 */
function WeatherSystem({ weather, timeOfDay }) {
  const weatherConfigs = {
    sunny: { clouds: 0, rain: 0, fog: 0 },
    cloudy: { clouds: 0.5, rain: 0, fog: 0.2 },
    rainy: { clouds: 0.8, rain: 0.6, fog: 0.3 },
    foggy: { clouds: 0.3, rain: 0, fog: 0.7 },
    stormy: { clouds: 1.0, rain: 0.9, fog: 0.4 }
  };

  const config = weatherConfigs[weather] || weatherConfigs.sunny;

  return (
    <>
      {config.clouds > 0 && (
        <Cloud position={[0, 5, 0]} opacity={config.clouds} />
      )}
      {config.rain > 0 && (
        <mesh>
          <boxGeometry args={[50, 50, 50]} />
          <meshStandardMaterial transparent opacity={config.rain * 0.3} color={0x4a90e2} />
        </mesh>
      )}
    </>
  );
}

/**
 * Environment Scene
 */
function EnvironmentScene({ environmentConfig, weather, timeOfDay }) {
  const sceneRef = useRef();

  // Lighting based on time of day
  const lightingConfigs = {
    day: { ambient: 0.7, directional: 1.0, color: 0xffffff },
    sunset: { ambient: 0.5, directional: 0.8, color: 0xff6b35 },
    night: { ambient: 0.2, directional: 0.3, color: 0x4a90e2 },
    dawn: { ambient: 0.4, directional: 0.6, color: 0xffd700 }
  };

  const lighting = lightingConfigs[timeOfDay] || lightingConfigs.day;

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={lighting.ambient} color={lighting.color} />
      <directionalLight
        position={environmentConfig.directionalLight.position}
        intensity={lighting.directional * environmentConfig.directionalLight.intensity}
        color={environmentConfig.directionalLight.color}
      />
      <pointLight position={[0, 5, 0]} intensity={0.5} color={environmentConfig.ambientLight.color} />

      {/* Sky/Environment */}
      {timeOfDay === 'night' ? (
        <Stars radius={100} depth={50} count={5000} factor={4} fade speed={1} />
      ) : (
        <Sky
          sunPosition={[10, 10, 5]}
          inclination={0.49}
          azimuth={0.25}
          turbidity={weather === 'stormy' ? 10 : 3}
        />
      )}

      {/* Weather Effects */}
      <WeatherSystem weather={weather} timeOfDay={timeOfDay} />

      {/* Ground Plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color={environmentConfig.theme === 'nature' ? 0x90ee90 : 0x333333} />
      </mesh>

      {/* Particle System */}
      <ParticleSystem type={environmentConfig.particles} />

      {/* Environment Name Display */}
      <Text
        position={[0, 8, 0]}
        fontSize={1}
        color={environmentConfig.ambientLight.color}
        anchorX="center"
        anchorY="middle"
      >
        {environmentConfig.name}
      </Text>
    </>
  );
}

/**
 * Main Virtual Environment Component
 */
export default function VirtualEnvironment({ environmentId, onEnvironmentChange, className = '' }) {
  const { currentEnvironment, setEnvironment, getThemeColors } = useVirtualEnvironment();

  useEffect(() => {
    if (environmentId && environmentId !== currentEnvironment?.id) {
      setEnvironment(environmentId);
    }
  }, [environmentId, currentEnvironment, setEnvironment]);

  if (!currentEnvironment) {
    return (
      <div className={`virtual-environment-container ${className}`}>
        <div className="environment-placeholder">
          <p>Select an environment to begin</p>
        </div>
      </div>
    );
  }

  const themeColors = getThemeColors();

  return (
    <div 
      className={`virtual-environment-container ${className}`}
      style={{
        '--theme-primary': themeColors.primary,
        '--theme-secondary': themeColors.secondary,
        '--theme-accent': themeColors.accent
      }}
    >
      <Canvas
        camera={{ position: [0, 5, 10], fov: 75 }}
        shadows
        gl={{ antialias: true }}
      >
        <Suspense fallback={null}>
          <EnvironmentScene
            environmentConfig={currentEnvironment.config}
            weather={currentEnvironment.weather}
            timeOfDay={currentEnvironment.timeOfDay}
          />
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={5}
            maxDistance={50}
          />
        </Suspense>
      </Canvas>
      
      <div className="environment-controls">
        <div className="environment-info">
          <h3>{currentEnvironment.config.name}</h3>
          <p>{currentEnvironment.config.description}</p>
        </div>
      </div>
    </div>
  );
}

