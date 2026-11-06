
"use strict";

const $ = id => document.getElementById(id);

const fmt = n => Number(n.toFixed(6));

const identity = n => Array.from({length: n}, (_, i) =>
  Array.from({length: n}, (_, j) => i === j ? 1 : 0)
);

const titles = {
  1: "Diagonalization of a Symmetric Matrix",
  2: "Gaussian Elimination and LU Decomposition",
  3: "Null Space Verification",
  4: "Three-State Markov Chain Simulation",
  5: "Matrix Representation of a Linear Transformation",
  6: "Computing the Null Space",
  7: "Gradient Calculation",
  8: "Gradient Vector Field Visualization"
};

const descriptions = {
  1: "Uses the Jacobi eigenvalue algorithm to diagonalize a real symmetric matrix.",
  2: "Solves the same linear system using Gaussian elimination and LU decomposition.",
  3: "Checks whether Ax equals the zero vector.",
  4: "Simulates transitions between three states using a probability matrix.",
  5: "Builds the matrix using the images of the standard basis vectors.",
  6: "Uses reduced row-echelon form to find a basis for the null space.",
  7: "Calculates the partial derivatives at the point you enter.",
  8: "Draws arrows representing the gradient at points on a grid."
};

function output(text, error = false) {
  $("result").textContent = text;
  $("result").classList.toggle("error", error);
}

function showProgram(n) {
  $("programPanel").classList.remove("hidden");

  $("questionLabel").textContent = `QUESTION ${n}`;
  $("programTitle").textContent = titles[n];
  $("programDescription").textContent = descriptions[n];

  $("gradientInputs").classList.toggle("hidden", n !== 7);
  $("canvasWrap").classList.toggle("hidden", n !== 8);

  document.querySelectorAll(".question-button").forEach(b =>
    b.classList.toggle("active", Number(b.dataset.question) === n)
  );

  runQuestion(n);

  $("programPanel").scrollIntoView({
    behavior: "smooth",
    block: "start"
  });
}

// QUESTION 1: JACOBI DIAGONALIZATION

function jacobi(A) {
  const n = A.length;
  const D = A.map(r => r.slice());
  const Q = identity(n);

  for (let it = 0; it < 100 * n * n; it++) {
    let p = 0, q = 1, max = 0;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(D[i][j]) > max) {
          max = Math.abs(D[i][j]);
          p = i;
          q = j;
        }
      }
    }

    if (max < 1e-10) break;

    const theta = 0.5 * Math.atan2(
      2 * D[p][q],
      D[q][q] - D[p][p]
    );

    const c = Math.cos(theta);
    const s = Math.sin(theta);

    const app = c*c*D[p][p] - 2*s*c*D[p][q] + s*s*D[q][q];
    const aqq = s*s*D[p][p] + 2*s*c*D[p][q] + c*c*D[q][q];

    for (let k = 0; k < n; k++) {
      if (k !== p && k !== q) {
        const x = D[k][p];
        const y = D[k][q];

        D[k][p] = D[p][k] = c*x - s*y;
        D[k][q] = D[q][k] = s*x + c*y;
      }
    }

    D[p][p] = app;
    D[q][q] = aqq;
    D[p][q] = D[q][p] = 0;

    for (let k = 0; k < n; k++) {
      const x = Q[k][p];
      const y = Q[k][q];

      Q[k][p] = c*x - s*y;
      Q[k][q] = s*x + c*y;
    }
  }

  return {D, Q};
}

// QUESTION 2: GAUSSIAN ELIMINATION

function gaussian(A, b) {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);

  for (let k = 0; k < n; k++) {
    let p = k;

    for (let i = k + 1; i < n; i++) {
      if (Math.abs(M[i][k]) > Math.abs(M[p][k])) {
        p = i;
      }
    }

    [M[k], M[p]] = [M[p], M[k]];

    if (Math.abs(M[k][k]) < 1e-12) {
      throw Error("Singular matrix.");
    }

    for (let i = k + 1; i < n; i++) {
      const f = M[i][k] / M[k][k];

      for (let j = k; j <= n; j++) {
        M[i][j] -= f * M[k][j];
      }
    }
  }

  const x = Array(n).fill(0);

  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];

    for (let j = i + 1; j < n; j++) {
      s -= M[i][j] * x[j];
    }

    x[i] = s / M[i][i];
  }

  return x;
}

// LU DECOMPOSITION

function luSolve(A, b) {
  const n = b.length;
  const L = identity(n);
  const U = A.map(r => r.slice());

  for (let k = 0; k < n; k++) {
    if (Math.abs(U[k][k]) < 1e-12) {
      throw Error("LU factorization encountered a zero pivot.");
    }

    for (let i = k + 1; i < n; i++) {
      const f = U[i][k] / U[k][k];
      L[i][k] = f;

      for (let j = k; j < n; j++) {
        U[i][j] -= f * U[k][j];
      }
    }
  }

  const y = Array(n).fill(0);
  const x = Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    y[i] = b[i];

    for (let j = 0; j < i; j++) {
      y[i] -= L[i][j] * y[j];
    }
  }

  for (let i = n - 1; i >= 0; i--) {
    x[i] = y[i];

    for (let j = i + 1; j < n; j++) {
      x[i] -= U[i][j] * x[j];
    }

    x[i] /= U[i][i];
  }

  return {x, L, U};
}

// QUESTION 3: NULL SPACE VERIFICATION

function matVec(A, x) {
  return A.map(r =>
    r.reduce((s, v, j) => s + v * x[j], 0)
  );
}

// QUESTION 4: MARKOV CHAIN

function markov(P, start, steps) {
  let state = start;
  const seq = [state];

  for (let k = 0; k < steps; k++) {
    const r = Math.random();
    let sum = 0;

    for (let j = 0; j < P.length; j++) {
      sum += P[state][j];

      if (r < sum || j === P.length - 1) {
        state = j;
        break;
      }
    }

    seq.push(state);
  }

  return seq;
}

// QUESTION 5: LINEAR TRANSFORMATION

function transform(x, y) {
  return [2*x + y, x + 3*y];
}

// QUESTION 6: NULL SPACE USING RREF

function nullSpace(A) {
  const R = A.map(r => r.slice());
  const m = R.length;
  const n = R[0].length;
  const pivots = [];

  let row = 0;

  for (let col = 0; col < n && row < m; col++) {
    let p = row;

    for (let i = row + 1; i < m; i++) {
      if (Math.abs(R[i][col]) > Math.abs(R[p][col])) {
        p = i;
      }
    }

    if (Math.abs(R[p][col]) < 1e-10) continue;

    [R[row], R[p]] = [R[p], R[row]];

    const d = R[row][col];

    for (let j = 0; j < n; j++) {
      R[row][j] /= d;
    }

    for (let i = 0; i < m; i++) {
      if (i !== row) {
        const f = R[i][col];

        for (let j = 0; j < n; j++) {
          R[i][j] -= f * R[row][j];
        }
      }
    }

    pivots.push(col);
    row++;
  }

  const free = Array.from({length: n}, (_, i) => i)
    .filter(c => !pivots.includes(c));

  const basis = free.map(f => {
    const v = Array(n).fill(0);
    v[f] = 1;

    pivots.forEach((p, i) => {
      v[p] = -R[i][f];
    });

    return v;
  });

  return {R, pivots, free, basis};
}

// QUESTION 7: GRADIENT

function gradient(x, y) {
  return [3*x*x + 2*y, 2*x + 2*y];
}

// QUESTION 8: GRADIENT VECTOR FIELD

function drawField() {
  const c = $("gradientCanvas");
  const ctx = c.getContext("2d");

  const w = c.width;
  const h = c.height;
  const s = 42;
  const ox = w / 2;
  const oy = h / 2;

  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "#e2e7ee";
  ctx.lineWidth = 1;

  for (let x = 0; x <= w; x += s) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }

  for (let y = 0; y <= h; y += s) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#555";
  ctx.beginPath();
  ctx.moveTo(0, oy);
  ctx.lineTo(w, oy);
  ctx.moveTo(ox, 0);
  ctx.lineTo(ox, h);
  ctx.stroke();

  ctx.fillStyle = "#333";
  ctx.font = "13px Arial";
  ctx.fillText("x", w - 18, oy - 8);
  ctx.fillText("y", ox + 8, 15);

  for (let x = -7; x <= 7; x++) {
    for (let y = -4; y <= 4; y++) {
      const [gx, gy] = gradient(x / 2, y / 2);
      const mag = Math.hypot(gx, gy);

      if (mag < 1e-8) continue;

      const len = Math.min(23, 5 + mag * 0.35);
      const dx = gx / mag * len;
      const dy = -gy / mag * len;

      const px = ox + x / 2 * s;
      const py = oy - y / 2 * s;
      const a = Math.atan2(dy, dx);

      ctx.strokeStyle = "#003366";
      ctx.fillStyle = "#003366";
      ctx.lineWidth = 1.8;

      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + dx, py + dy);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(px + dx, py + dy);

      ctx.lineTo(
        px + dx - 6 * Math.cos(a - 0.48),
        py + dy - 6 * Math.sin(a - 0.48)
      );

      ctx.lineTo(
        px + dx - 6 * Math.cos(a + 0.48),
        py + dy - 6 * Math.sin(a + 0.48)
      );

      ctx.closePath();
      ctx.fill();
    }
  }
}

// RUN SELECTED QUESTION

function runQuestion(n) {
  try {
    if (n === 1) {
      const A = [[4,1,1],[1,3,0],[1,0,2]];
      const {D, Q} = jacobi(A);

      return output(
        `Input matrix A:\n${JSON.stringify(A)}\n\n` +
        `Diagonal matrix D:\n${JSON.stringify(
          D.map((r,i) => r.map((v,j) => i === j ? fmt(v) : 0))
        )}\n\n` +
        `Orthogonal matrix Q:\n${JSON.stringify(Q.map(r => r.map(fmt)))}\n\n` +
        `A = Q D Qᵀ (within numerical rounding).`
      );
    }

    if (n === 2) {
      const A = [[3,1,-1],[2,4,1],[-1,2,5]];
      const b = [4,1,1];

      const g = gaussian(A, b);
      const lu = luSolve(A, b);

      return output(
        `A = ${JSON.stringify(A)}\n` +
        `b = ${JSON.stringify(b)}\n\n` +
        `Gaussian elimination solution:\n${JSON.stringify(g.map(fmt))}\n\n` +
        `LU decomposition solution:\n${JSON.stringify(lu.x.map(fmt))}\n\n` +
        `L:\n${JSON.stringify(lu.L.map(r => r.map(fmt)))}\n\n` +
        `U:\n${JSON.stringify(lu.U.map(r => r.map(fmt)))}\n\n` +
        `Both methods give the same solution (within rounding).`
      );
    }

    if (n === 3) {
      const A = [[1,2,1],[2,4,2]];
      const x = [1,0,-1];
      const ax = matVec(A, x);

      const ok = ax.every(v => Math.abs(v) < 1e-9);

      return output(
        `A = ${JSON.stringify(A)}\n` +
        `x = ${JSON.stringify(x)}\n\n` +
        `Ax = ${JSON.stringify(ax)}\n\n` +
        `${ok
          ? "The vector belongs to the null space because Ax = 0."
          : "The vector does not belong to the null space."}`
      );
    }

    if (n === 4) {
      const P = [
        [.5,.3,.2],
        [.2,.6,.2],
        [.3,.2,.5]
      ];

      const names = ["A","B","C"];
      const seq = markov(P, 0, 20).map(i => names[i]);

      return output(
        `Transition matrix:\n${JSON.stringify(P)}\n\n` +
        `Starting state: A\n` +
        `20 transitions:\n${seq.join(" → ")}\n\n` +
        `The sequence changes on each run because transitions are random.`
      );
    }

    if (n === 5) {
      const a = transform(1, 0);
      const b = transform(0, 1);

      const M = [
        [a[0], b[0]],
        [a[1], b[1]]
      ];

      return output(
        `T(1,0) = ${JSON.stringify(a)}\n` +
        `T(0,1) = ${JSON.stringify(b)}\n\n` +
        `Matrix representation:\n${JSON.stringify(M)}\n\n` +
        `T(x,y) = (2x+y, x+3y).`
      );
    }

    if (n === 6) {
      const A = [[1,2,1],[2,4,2]];
      const r = nullSpace(A);

      return output(
        `A = ${JSON.stringify(A)}\n\n` +
        `RREF(A):\n${JSON.stringify(r.R.map(row => row.map(fmt)))}\n\n` +
        `Pivot columns: ${r.pivots.map(v => v+1).join(", ")}\n` +
        `Free columns: ${r.free.map(v => v+1).join(", ")}\n\n` +
        `Basis for N(A):\n` +
        `${r.basis.map(v => JSON.stringify(v.map(fmt))).join("\n")}\n\n` +
        `Nullity = ${r.basis.length}`
      );
    }

    if (n === 7) {
      $("gradientInputs").classList.remove("hidden");

      const x = Number($("gradientX").value);
      const y = Number($("gradientY").value);

      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return output("Enter valid numeric values for x and y.", true);
      }

      const [gx, gy] = gradient(x, y);

      return output(
        `f(x,y) = x³ + 2xy + y²\n` +
        `At (${x}, ${y}):\n` +
        `∂f/∂x = 3x² + 2y = ${fmt(gx)}\n` +
        `∂f/∂y = 2x + 2y = ${fmt(gy)}\n\n` +
        `Gradient = (${fmt(gx)}, ${fmt(gy)})`
      );
    }

    if (n === 8) {
      drawField();

      return output(
        "Gradient vector field drawn for ∇f(x,y) = " +
        "(3x² + 2y, 2x + 2y). Each arrow indicates " +
        "the direction of steepest increase."
      );
    }

  } catch (e) {
    output("Unable to run program: " + e.message, true);
  }
}

// STUDENT DETAILS FORM

$("detailsForm").addEventListener("submit", e => {
  e.preventDefault();

  const name = $("studentName").value.trim();
  const roll = $("registerNumber").value.trim();

  if (!name || !roll) return;

  $("displayName").textContent = name;

  $("studentSummary").textContent =
    `Register No: ${roll} | ${$("course").value} | Section: ${$("section").value}`;

  $("detailsPage").classList.add("hidden");
  $("assignmentPage").classList.remove("hidden");

  try {
    localStorage.setItem("b03Student", JSON.stringify({
      name,
      roll,
      course: $("course").value,
      section: $("section").value,
      faculty: $("faculty").value
    }));
  } catch (_) {}

  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
});

// QUESTION BUTTONS

document.querySelectorAll(".question-button").forEach(btn => {
  btn.addEventListener("click", () => {
    showProgram(Number(btn.dataset.question));
  });
});

// EDIT STUDENT DETAILS

$("editDetails").addEventListener("click", () => {
  $("assignmentPage").classList.add("hidden");
  $("detailsPage").classList.remove("hidden");
});

$("backToDetails").addEventListener("click", () => {
  $("assignmentPage").classList.add("hidden");
  $("detailsPage").classList.remove("hidden");
});

// CLEAR OUTPUT

$("clearOutput").addEventListener("click", () => {
  $("programPanel").classList.add("hidden");

  document.querySelectorAll(".question-button").forEach(b =>
    b.classList.remove("active")
  );
});

// CALCULATE GRADIENT

$("calculateGradient").addEventListener("click", () => {
  runQuestion(7);
});

// PRINT PAGE

$("printPage").addEventListener("click", () => {
  window.print();
});

// LOAD SAVED STUDENT DETAILS

try {
  const saved = JSON.parse(
    localStorage.getItem("b03Student") || "null"
  );

  if (saved) {
    $("studentName").value = saved.name || "";
    $("registerNumber").value = saved.roll || "";
    $("course").value = saved.course || "";
    $("section").value = saved.section || "";
    $("faculty").value = saved.faculty || "";
  }
} catch (_) {}
