const fs = require('fs');
const wav = require('node-wav');
const Meyda = require('meyda');

const audioPath = process.argv[2];
const rangesPath = process.argv[3];
const outputPath = process.argv[4];

if (!audioPath || !rangesPath || !outputPath) {
  console.error("Usage: node extract_js.js <audioPath> <rangesPath> <outputPath>");
  process.exit(1);
}

const buffer = fs.readFileSync(audioPath);
const result = wav.decode(buffer);
const y = result.channelData[0]; 
const sr = result.sampleRate; // This will be 16000

// --- THE CRITICAL FIXES ---
const FRAME_SIZE = 1024;
const HOP_SIZE = 512;

Meyda.bufferSize = FRAME_SIZE; 
Meyda.sampleRate = sr;  // Forces Meyda to align its math to 16kHz!
Meyda.melBands = 40;    // Increases frequency resolution to catch the breath hiss
// --------------------------

const ranges = JSON.parse(fs.readFileSync(rangesPath, 'utf8'));
const features = [];

for (let i = 0; i < ranges.length; i++) {
  const [start, end] = ranges[i];
  const window = y.slice(start, end);
  
  if (window.length !== 8192) {
    features.push(null);
    continue;
  }

  let mfccSum = new Array(13).fill(0);
  let centSum = 0;
  let zcrSum = 0;
  let rmsSum = 0;
  let frameCount = 0;

  for (let j = 0; j <= window.length - FRAME_SIZE; j += HOP_SIZE) {
    const frame = window.slice(j, j + FRAME_SIZE);
    
    try {
      const extracted = Meyda.extract(['mfcc', 'spectralCentroid', 'zcr', 'rms'], frame);
      if (extracted) {
        for (let k = 0; k < 13; k++) mfccSum[k] += extracted.mfcc[k];
        centSum += extracted.spectralCentroid;
        zcrSum += extracted.zcr;
        rmsSum += extracted.rms;
        frameCount++;
      }
    } catch (e) {
      // Ignore frame errors
    }
  }

  if (frameCount > 0) {
    const featStack = [
      ...mfccSum.map(val => val / frameCount),
      centSum / frameCount,
      zcrSum / frameCount,
      rmsSum / frameCount
    ];
    features.push(featStack);
  } else {
    features.push(null);
  }
}

fs.writeFileSync(outputPath, JSON.stringify(features));