/**
 * Virtual Environment Service
 * Manages 3D environments, themes, and immersive experiences
 */

class VirtualEnvironmentService {
  constructor() {
    this.currentEnvironment = null;
    this.environments = {
      futuristic_race_track: {
        name: "Futuristic Race Track",
        description: "High-speed productivity racing environment",
        theme: "cyberpunk",
        skybox: "/environments/race_track_skybox.jpg",
        ambientLight: { color: 0x00ffff, intensity: 0.5 },
        directionalLight: { color: 0xffffff, intensity: 1, position: [10, 10, 5] },
        particles: "speed_lines",
        audio: "energetic_electronic"
      },
      mysterious_library: {
        name: "Mysterious Library",
        description: "Ancient knowledge and wisdom",
        theme: "mystical",
        skybox: "/environments/library_skybox.jpg",
        ambientLight: { color: 0xffd700, intensity: 0.4 },
        directionalLight: { color: 0xffaa00, intensity: 0.8, position: [0, 10, 0] },
        particles: "glowing_puzzles",
        audio: "ambient_thoughtful"
      },
      artistic_studio: {
        name: "Artistic Studio",
        description: "Creative workspace for inspiration",
        theme: "creative",
        skybox: "/environments/studio_skybox.jpg",
        ambientLight: { color: 0xff6b9d, intensity: 0.6 },
        directionalLight: { color: 0xffffff, intensity: 1, position: [5, 10, 5] },
        particles: "color_splashes",
        audio: "inspiring_acoustic"
      },
      cyber_code_space: {
        name: "Cyber Code Space",
        description: "Digital realm for coding challenges",
        theme: "tech",
        skybox: "/environments/code_space_skybox.jpg",
        ambientLight: { color: 0x00ff00, intensity: 0.5 },
        directionalLight: { color: 0x00ffff, intensity: 1, position: [0, 10, 0] },
        particles: "matrix_rain",
        audio: "tech_ambient"
      },
      serene_garden: {
        name: "Serene Garden",
        description: "Peaceful environment for focused work",
        theme: "nature",
        skybox: "/environments/garden_skybox.jpg",
        ambientLight: { color: 0x87ceeb, intensity: 0.7 },
        directionalLight: { color: 0xffffff, intensity: 0.9, position: [5, 10, 5] },
        particles: "leaves",
        audio: "nature_ambient"
      },
      space_station: {
        name: "Space Station",
        description: "Zero-gravity productivity zone",
        theme: "sci-fi",
        skybox: "/environments/space_skybox.jpg",
        ambientLight: { color: 0x4a90e2, intensity: 0.4 },
        directionalLight: { color: 0xffffff, intensity: 1.2, position: [0, 20, 0] },
        particles: "stars",
        audio: "space_ambient"
      },
      medieval_castle: {
        name: "Medieval Castle",
        description: "Epic quest environment",
        theme: "fantasy",
        skybox: "/environments/castle_skybox.jpg",
        ambientLight: { color: 0xff6b35, intensity: 0.5 },
        directionalLight: { color: 0xffaa44, intensity: 0.8, position: [10, 15, 5] },
        particles: "magic_sparks",
        audio: "epic_orchestral"
      },
      underwater_lab: {
        name: "Underwater Lab",
        description: "Deep focus environment",
        theme: "ocean",
        skybox: "/environments/ocean_skybox.jpg",
        ambientLight: { color: 0x0066cc, intensity: 0.6 },
        directionalLight: { color: 0x00ccff, intensity: 0.7, position: [0, 10, 10] },
        particles: "bubbles",
        audio: "underwater_ambient"
      },
      neon_city: {
        name: "Neon City",
        description: "Urban productivity landscape",
        theme: "urban",
        skybox: "/environments/city_skybox.jpg",
        ambientLight: { color: 0xff00ff, intensity: 0.5 },
        directionalLight: { color: 0xffffff, intensity: 1, position: [10, 20, 10] },
        particles: "neon_particles",
        audio: "urban_beat"
      },
      zen_temple: {
        name: "Zen Temple",
        description: "Mindful productivity space",
        theme: "zen",
        skybox: "/environments/temple_skybox.jpg",
        ambientLight: { color: 0xffd700, intensity: 0.5 },
        directionalLight: { color: 0xffffff, intensity: 0.7, position: [0, 10, 0] },
        particles: "cherry_blossoms",
        audio: "zen_meditation"
      }
    };
    this.weatherSystems = {
      sunny: { clouds: 0, rain: 0, fog: 0 },
      cloudy: { clouds: 0.5, rain: 0, fog: 0.2 },
      rainy: { clouds: 0.8, rain: 0.6, fog: 0.3 },
      foggy: { clouds: 0.3, rain: 0, fog: 0.7 },
      stormy: { clouds: 1.0, rain: 0.9, fog: 0.4 }
    };
    this.timeOfDay = "day"; // day, sunset, night, dawn
  }

  /**
   * Get available environments
   */
  getAvailableEnvironments() {
    return Object.keys(this.environments).map(key => ({
      id: key,
      ...this.environments[key]
    }));
  }

  /**
   * Set current environment
   */
  setEnvironment(environmentId, options = {}) {
    if (!this.environments[environmentId]) {
      console.warn(`Environment ${environmentId} not found`);
      return null;
    }

    this.currentEnvironment = {
      id: environmentId,
      config: this.environments[environmentId],
      weather: options.weather || "sunny",
      timeOfDay: options.timeOfDay || this.timeOfDay,
      customizations: options.customizations || {}
    };

    return this.currentEnvironment;
  }

  /**
   * Get current environment
   */
  getCurrentEnvironment() {
    return this.currentEnvironment;
  }

  /**
   * Update weather
   */
  setWeather(weatherType) {
    if (!this.weatherSystems[weatherType]) {
      console.warn(`Weather type ${weatherType} not found`);
      return;
    }

    if (this.currentEnvironment) {
      this.currentEnvironment.weather = weatherType;
    }
  }

  /**
   * Update time of day
   */
  setTimeOfDay(timeOfDay) {
    this.timeOfDay = timeOfDay;
    if (this.currentEnvironment) {
      this.currentEnvironment.timeOfDay = timeOfDay;
    }
  }

  /**
   * Get environment bonus based on challenge type
   */
  getEnvironmentBonus(challengeType) {
    if (!this.currentEnvironment) return 1.0;

    const bonuses = {
      futuristic_race_track: { time_attack: 1.2, coding_kata: 1.1 },
      mysterious_library: { puzzle: 1.3, creative_sprint: 1.1 },
      artistic_studio: { creative_sprint: 1.3, puzzle: 1.1 },
      cyber_code_space: { coding_kata: 1.3, time_attack: 1.1 },
      serene_garden: { puzzle: 1.2, creative_sprint: 1.1 },
      space_station: { time_attack: 1.2, coding_kata: 1.1 },
      medieval_castle: { puzzle: 1.2, creative_sprint: 1.1 },
      underwater_lab: { puzzle: 1.2, coding_kata: 1.1 },
      neon_city: { time_attack: 1.2, coding_kata: 1.1 },
      zen_temple: { puzzle: 1.3, creative_sprint: 1.2 }
    };

    const envBonuses = bonuses[this.currentEnvironment.id] || {};
    return envBonuses[challengeType] || 1.0;
  }

  /**
   * Unlock environment (for progression)
   */
  unlockEnvironment(environmentId) {
    // This would typically check user progress/achievements
    return true;
  }

  /**
   * Get environment theme colors
   */
  getThemeColors() {
    if (!this.currentEnvironment) {
      return { primary: "#6366f1", secondary: "#8b5cf6", accent: "#ec4899" };
    }

    const themeColors = {
      cyberpunk: { primary: "#00ffff", secondary: "#ff00ff", accent: "#ffff00" },
      mystical: { primary: "#ffd700", secondary: "#ff8c00", accent: "#ff6347" },
      creative: { primary: "#ff6b9d", secondary: "#c44569", accent: "#f8b500" },
      tech: { primary: "#00ff00", secondary: "#00ffff", accent: "#0000ff" },
      nature: { primary: "#87ceeb", secondary: "#90ee90", accent: "#ffd700" },
      "sci-fi": { primary: "#4a90e2", secondary: "#7b68ee", accent: "#00ced1" },
      fantasy: { primary: "#ff6b35", secondary: "#f7931e", accent: "#ffd700" },
      ocean: { primary: "#0066cc", secondary: "#00ccff", accent: "#00ffcc" },
      urban: { primary: "#ff00ff", secondary: "#00ffff", accent: "#ffff00" },
      zen: { primary: "#ffd700", secondary: "#ffa500", accent: "#ff6347" }
    };

    return themeColors[this.currentEnvironment.config.theme] || themeColors.cyberpunk;
  }
}

// Export singleton instance
export default new VirtualEnvironmentService();

