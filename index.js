const fs = require('fs');
// const b = opencv._doMalloc(160000);
const PNG = require('pngjs').PNG;

function Module(moduleName, moduleFn) {};
Module.print = (text) => { console.log(text); };
Module.printErr = (text) => { console.warn(text); };
Module.onRuntimeInitialized = () => {
  // console.log('loaded');
  run();
};
globalThis.Module = Module;
const opencv = require('./build_wasm/bin/opencv.js');

function run() {

class Allocator {
  constructor() {
    this.offsets = [];
  }
  alloc(constructor, size) {
    console.log('alloc', size * constructor.BYTES_PER_ELEMENT);
    const offset = opencv._doMalloc(size * constructor.BYTES_PER_ELEMENT);
    const b = new constructor(opencv.HEAP8.buffer, opencv.HEAP8.byteOffset + offset, size);
    b.offset = offset;
    this.offsets.push(offset);
    return b;
  }
  freeAll() {
    for (let i = 0; i < this.offsets.length; i++) {
      opencv._doFree(this.offsets[i]);
    }
    this.offsets.length = 0;
  }
}

fs.createReadStream('./qr.png')
  .pipe(
    new PNG({
      filterType: 4,
    })
  )
  .on('parsed', function() {
setTimeout(() => {
    const {
      width,
      height,
      data: imgDataData,
    } = this;
    console.log('parsed', width, height, imgDataData.length);

    const viewMatrixInverseData = Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const projectionMatrixInverseData = Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

    const allocator = new Allocator();

    const imgData = allocator.alloc(Uint8Array, imgDataData.length);
    imgData.set(imgDataData);

    const viewMatrixInverse = allocator.alloc(Float32Array, viewMatrixInverseData.length);
    viewMatrixInverse.set(viewMatrixInverseData);

    const projectionMatrixInverse = allocator.alloc(Float32Array, projectionMatrixInverseData.length);
    projectionMatrixInverse.set(projectionMatrixInverseData);

    const qrCodes = allocator.alloc(Float32Array, 1024 * Float32Array.BYTES_PER_ELEMENT);
    const qrCodeString = allocator.alloc(Uint8Array, 1024 * Float32Array.BYTES_PER_ELEMENT);

    const numQrCodes = allocator.alloc(Uint32Array, 1);
    numQrCodes[0] = 0;
    const qrCodeStringLength = allocator.alloc(Uint32Array, 1);
    qrCodeStringLength[0] = 0;

    console.log('get qr 1');

    opencv._doGetQr(
      width,
      height,
      imgData.offset,
      viewMatrixInverse.offset,
      projectionMatrixInverse.offset,
      qrCodes.offset,
      numQrCodes.offset,
      qrCodeString.offset,
      qrCodeStringLength.offset
    );

    const s = new TextDecoder().decode(qrCodeString.slice(0, qrCodeStringLength[0]));
    console.log('get qr 2', s, qrCodeStringLength[0], qrCodes, numQrCodes[0]);

    const arrayBuffer = new ArrayBuffer(
      Uint32Array.BYTES_PER_ELEMENT +
      numQrCodes[0] * 4 * Float32Array.BYTES_PER_ELEMENT
    );
    let index = 0;

    const outQrCodes = new Float32Array(arrayBuffer, index, numQrCodes[0] * 4);
    outQrCodes.set(new Float32Array(qrCodes.buffer, qrCodes.byteOffset, numQrCodes[0] * 4));
    index += Float32Array.BYTES_PER_ELEMENT * numQrCodes[0] * 4;

    allocator.freeAll();
});
  });

}