// Same formula as engine/queuesim/power.py (tests/test_power.py checks the constants).
// Two independent proportions, fixed horizon, two-sided 5%, power 80%.
const Z_ALPHA = 1.959963984540054;
const Z_BETA = 0.8416212335729143;

export function weeksNeeded(p0: number, rel: number, leadsPerWeek: number, share = 0.5): number {
  const p1 = Math.min(p0 * (1 + rel), 0.999999);
  const d = p1 - p0;
  if (p0 <= 0 || d === 0 || leadsPerWeek <= 0 || share <= 0 || share >= 1) return Infinity;
  const z = Z_ALPHA + Z_BETA;
  const nTotal = (z * z * ((p1 * (1 - p1)) / share + (p0 * (1 - p0)) / (1 - share))) / (d * d);
  return nTotal / leadsPerWeek;
}
