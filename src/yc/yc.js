// Yield-curve PCA in the browser, mirrored from yc/pca.py (tested against it through Node).
const YC = (() => {
  // Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations.
  function eigSym(A) {
    const n = A.length, a = A.map((r) => r.slice());
    const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
        }
      }
    }
    return { values: a.map((r, i) => r[i]), vectors: v };
  }
  // PCA of the columns of X (rows = months), signs fixed as in pca.py.
  function pca(X) {
    const T = X.length, n = X[0].length;
    const mean = Array.from({ length: n }, (_, j) => X.reduce((s, r) => s + r[j], 0) / T);
    const C = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) =>
      X.reduce((s, r) => s + (r[i] - mean[i]) * (r[j] - mean[j]), 0) / (T - 1)));
    const { values, vectors } = eigSym(C);
    const order = values.map((v, i) => [v, i]).sort((x, y) => y[0] - x[0]).map(([, i]) => i);
    const vals = order.map((i) => values[i]);
    const L = Array.from({ length: n }, (_, r) => order.map((i) => vectors[r][i]));
    const flip = (k) => { for (let r = 0; r < n; r++) L[r][k] *= -1; };
    if (L.reduce((s, r) => s + r[0], 0) < 0) flip(0);
    if (n > 1 && L[n - 1][1] < L[0][1]) flip(1);
    if (n > 2 && L[Math.floor(n / 2)][2] > (L[0][2] + L[n - 1][2]) / 2) flip(2);
    const tot = vals.reduce((s, v) => s + v, 0);
    const scores = X.map((r) => order.map((_, k) => r.reduce((s, x, j) => s + (x - mean[j]) * L[j][k], 0)));
    return { values: vals, share: vals.map((v) => v / tot), loadings: L, scores, mean };
  }
  return { eigSym, pca };
})();
if (typeof module !== "undefined") module.exports = YC;
