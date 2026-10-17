// Scene modules loaded by the host, relative to js/scenes/. A module that is missing or throws
// while loading is skipped (and reported in the console / snapshot), so one broken scene never
// takes the app down.
export const MODULES = [
  'moon.js',      // reference implementation of the scene contract
  'wormhole.js',
  'neutron.js',
  'stars.js',
  'rocky.js',
  'giants.js',
  'galaxy.js',
  'nebulae.js',
  'small.js',
];
