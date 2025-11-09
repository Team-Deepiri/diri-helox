import { useState, useEffect, useCallback } from 'react';
import virtualEnvironmentService from '../services/virtualEnvironmentService';

/**
 * Hook for managing virtual environment state
 */
export function useVirtualEnvironment() {
  const [currentEnvironment, setCurrentEnvironmentState] = useState(
    virtualEnvironmentService.getCurrentEnvironment()
  );

  const setEnvironment = useCallback((environmentId, options = {}) => {
    const env = virtualEnvironmentService.setEnvironment(environmentId, options);
    setCurrentEnvironmentState(env);
    return env;
  }, []);

  const setWeather = useCallback((weatherType) => {
    virtualEnvironmentService.setWeather(weatherType);
    setCurrentEnvironmentState(virtualEnvironmentService.getCurrentEnvironment());
  }, []);

  const setTimeOfDay = useCallback((timeOfDay) => {
    virtualEnvironmentService.setTimeOfDay(timeOfDay);
    setCurrentEnvironmentState(virtualEnvironmentService.getCurrentEnvironment());
  }, []);

  const getAvailableEnvironments = useCallback(() => {
    return virtualEnvironmentService.getAvailableEnvironments();
  }, []);

  const getEnvironmentBonus = useCallback((challengeType) => {
    return virtualEnvironmentService.getEnvironmentBonus(challengeType);
  }, []);

  const getThemeColors = useCallback(() => {
    return virtualEnvironmentService.getThemeColors();
  }, []);

  return {
    currentEnvironment,
    setEnvironment,
    setWeather,
    setTimeOfDay,
    getAvailableEnvironments,
    getEnvironmentBonus,
    getThemeColors
  };
}

