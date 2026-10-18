// Scene modules loaded by the host, relative to js/scenes/. A module that is missing or throws
// while loading is skipped (and reported in the console / snapshot), so one broken scene never
// takes the app down. When two modules define the same scene id, the one listed first wins.
export const MODULES = [
  'moon.js',      // reference implementation of the scene contract
  'wormhole.js',
  'neutron.js',
  'stars.js',
  // one module per Solar System planet (these supersede the older entries in rocky.js / giants.js)
  'planets/mercury.js',
  'planets/venus.js',
  'planets/earth.js',
  'planets/mars.js',
  'planets/jupiter.js',
  'planets/saturn.js',
  'planets/uranus.js',
  'planets/neptune.js',
  'lava.js',
  'browndwarf.js',
  'galaxy.js',
  // one module per nebula
  'nebulae/pillars.js',
  'nebulae/ring.js',
  'nebulae/crab.js',
  'nebulae/horsehead.js',
  'small.js',
];
