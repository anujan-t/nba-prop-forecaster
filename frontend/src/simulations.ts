import { useCallback, useEffect, useState } from 'react';

/** How many games the simulator plays for every forecast and parlay. More runs shrink simulation noise only. */
export const SIMULATION_CHOICES = [100_000, 250_000, 500_000, 1_000_000] as const;
export const DEFAULT_SIMULATIONS = 250_000;
const storageKey = 'courtside.simulations';

const valid = (value: number) => (SIMULATION_CHOICES as readonly number[]).includes(value);

export function loadSimulations(): number {
  try {
    const value = Number(window.localStorage.getItem(storageKey));
    return valid(value) ? value : DEFAULT_SIMULATIONS;
  } catch {
    return DEFAULT_SIMULATIONS;
  }
}

/** Worst-case simulation error of a probability, in percentage points (at 50%). */
export function simulationError(runs: number): number {
  return 100 * Math.sqrt(0.25 / runs);
}

export function useSimulations(): [number, (value: number) => void] {
  const [simulations, setSimulations] = useState(loadSimulations);
  useEffect(() => {
    const onStorage = (event: StorageEvent) => { if (event.key === storageKey) setSimulations(loadSimulations()); };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);
  const choose = useCallback((value: number) => {
    if (!valid(value)) return;
    setSimulations(value);
    try { window.localStorage.setItem(storageKey, String(value)); } catch { /* storage unavailable: the choice lasts for this page */ }
  }, []);
  return [simulations, choose];
}
