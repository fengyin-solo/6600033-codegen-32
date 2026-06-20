import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface MCScenario {
  id: string
  name: string
  description: string
  params: Record<string, number>
  category: string
}

export interface MCResult {
  scenario: string
  iterations: number
  estimate: number
  trueValue?: number
  error?: number
  samples: number[]
  convergence: number[]
}

export interface HypTestResult {
  testType: string
  statistic: number
  pValue: number
  significant: boolean
  alpha: number
  df?: number
}

export interface MOOSolution {
  weights: number[]
  expected_return: number
  risk: number
  sample_cost: number
  rank: number
  crowding_distance: number
  score?: number
}

export interface MOOResult {
  pareto_front: MOOSolution[]
  all_solutions: MOOSolution[]
  recommended: MOOSolution
  generations: number
  population_size: number
}

export interface MOOConfig {
  assets: number
  population_size: number
  generations: number
  expected_returns: number[]
  cov_matrix: number[][]
  cost_per_sample: number
  sample_size_base: number
  risk_weight: number
  return_weight: number
  cost_weight: number
  use_backend: boolean
  seed: number | null
}

function normalRandom(): number {
  let u = 0, v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}

function runMC(scenario: MCScenario, n: number): MCResult {
  const samples: number[] = []
  const convergence: number[] = []

  if (scenario.id === 'pi') {
    let inside = 0
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1
      if (x * x + y * y <= 1) inside++
      samples.push(x * x + y * y <= 1 ? 1 : 0)
      convergence.push((inside / (i + 1)) * 4)
    }
    const estimate = (inside / n) * 4
    return { scenario: 'pi', iterations: n, estimate, trueValue: Math.PI, error: Math.abs(estimate - Math.PI), samples, convergence }
  }
  if (scenario.id === 'brownian') {
    let pos = 0
    const dt = scenario.params.dt || 0.01
    for (let i = 0; i < n; i++) { pos += normalRandom() * Math.sqrt(dt); samples.push(pos) }
    convergence.push(...samples.slice(0, 200))
    return { scenario: 'brownian', iterations: n, estimate: pos, samples, convergence }
  }
  if (scenario.id === 'option') {
    const { S0 = 100, K = 105, r = 0.05, sigma = 0.2, T = 1 } = scenario.params
    let payoffSum = 0
    for (let i = 0; i < n; i++) {
      const ST = S0 * Math.exp((r - 0.5 * sigma * sigma) * T + sigma * Math.sqrt(T) * normalRandom())
      const p = Math.max(ST - K, 0); payoffSum += p; samples.push(p)
      if ((i + 1) % 50 === 0) convergence.push((payoffSum / (i + 1)) * Math.exp(-r * T))
    }
    return { scenario: 'option', iterations: n, estimate: (payoffSum / n) * Math.exp(-r * T), samples, convergence }
  }
  if (scenario.id === 'random_walk') {
    let pos = 0
    for (let i = 0; i < n; i++) { pos += Math.random() > 0.5 ? 1 : -1; samples.push(pos) }
    convergence.push(...samples.slice(0, 200))
    return { scenario: 'random_walk', iterations: n, estimate: pos, samples, convergence }
  }
  if (scenario.id === 'diffusion') {
    const { D = 1, dt = 0.01 } = scenario.params
    let x = 0, y = 0
    for (let i = 0; i < n; i++) {
      x += normalRandom() * Math.sqrt(2 * D * dt); y += normalRandom() * Math.sqrt(2 * D * dt)
      samples.push(Math.sqrt(x * x + y * y))
    }
    convergence.push(...samples.slice(0, 200))
    return { scenario: 'diffusion', iterations: n, estimate: Math.sqrt(x * x + y * y), samples, convergence }
  }
  // gambler
  const { p = 0.45, bankroll = 50, goal = 100 } = scenario.params
  let ruinCount = 0
  for (let i = 0; i < n; i++) {
    let money = bankroll
    let steps = 0
    while (money > 0 && money < goal && steps < 10000) { money += Math.random() < p ? 1 : -1; steps++ }
    if (money <= 0) ruinCount++
    samples.push(money <= 0 ? 0 : 1)
    convergence.push(ruinCount / (i + 1))
  }
  return { scenario: 'gambler', iterations: n, estimate: ruinCount / n, samples, convergence }
}

export const SCENARIOS: MCScenario[] = [
  { id: 'pi', name: '圆周率π估算', description: '随机投点估算π值，观察收敛过程', params: {}, category: '基础' },
  { id: 'brownian', name: '布朗运动模拟', description: '粒子热运动随机路径模拟', params: { dt: 0.01 }, category: '物理' },
  { id: 'option', name: '欧式期权定价', description: 'Black-Scholes期权价格蒙特卡洛估算', params: { S0: 100, K: 105, r: 0.05, sigma: 0.2, T: 1 }, category: '金融' },
  { id: 'random_walk', name: '随机游走', description: '一维离散随机游走轨迹模拟', params: {}, category: '基础' },
  { id: 'diffusion', name: '粒子扩散', description: '二维粒子随机扩散位移分析', params: { D: 1, dt: 0.01 }, category: '物理' },
  { id: 'gambler', name: '赌徒破产', description: '不利赌局下资金耗尽概率估算', params: { p: 0.45, bankroll: 50, goal: 100 }, category: '概率' }
]

function _mooNormalRandom(seeded?: SeededRng): number {
  if (seeded) return seeded.normal()
  let u = 0, v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}

class SeededRng {
  private state: number
  constructor(seed: number) { this.state = seed >>> 0 || 1 }
  next(): number {
    this.state = (1664525 * this.state + 1013904223) >>> 0
    return this.state / 0x100000000
  }
  normal(): number {
    let u = 0, v = 0
    while (u === 0) u = this.next()
    while (v === 0) v = this.next()
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
  }
}

function _generateDefaultReturns(n: number, rng: SeededRng | null): number[] {
  const r: number[] = []
  for (let i = 0; i < n; i++) {
    r.push(rng ? 0.03 + rng.next() * 0.22 : 0.03 + Math.random() * 0.22)
  }
  return r
}

function _generateDefaultCov(n: number, rng: SeededRng | null): number[][] {
  const A: number[][] = []
  for (let i = 0; i < n; i++) {
    A[i] = []
    for (let j = 0; j < n; j++) {
      A[i][j] = rng ? rng.normal() : _mooNormalRandom()
    }
  }
  const cov: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let k = 0; k < n; k++) s += A[i][k] * A[j][k]
      cov[i][j] = s / n
    }
  }
  let meanDiag = 0
  for (let i = 0; i < n; i++) meanDiag += cov[i][i]
  meanDiag /= n
  const scale = 0.04 / (meanDiag || 1)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) cov[i][j] *= scale
    cov[i][i] = Math.max(cov[i][i], 0.01)
  }
  return cov
}

function _dirichletWeights(n: number, rng: SeededRng | null): number[] {
  const w: number[] = []
  let sum = 0
  for (let i = 0; i < n; i++) {
    const x = -Math.log(rng ? rng.next() : Math.random())
    w.push(x); sum += x
  }
  return w.map(x => x / (sum || 1))
}

function _normalizeWeights(w: number[]): number[] {
  const nw = w.map(x => Math.max(0, x))
  const s = nw.reduce((a, b) => a + b, 0)
  if (s > 0) return nw.map(x => x / s)
  return new Array(nw.length).fill(1 / nw.length)
}

function _matVecSym(mat: number[][], v: number[]): number[] {
  const n = v.length, r = new Array(n).fill(0)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) r[i] += mat[i][j] * v[j]
  return r
}

function _evaluateWeights(w: number[], er: number[], cov: number[][], cps: number, ssb: number): [number, number, number] {
  let pRet = 0
  for (let i = 0; i < w.length; i++) pRet += w[i] * er[i]
  const mv = _matVecSym(cov, w)
  let quad = 0
  for (let i = 0; i < w.length; i++) quad += w[i] * mv[i]
  const pRisk = Math.sqrt(Math.max(0, quad))
  let sumSq = 1e-8
  for (let i = 0; i < w.length; i++) sumSq += w[i] * w[i]
  const diversity = 1 / sumSq
  const cost = cps * ssb * diversity
  return [pRet, pRisk, cost]
}

function _dominates(a: [number, number, number], b: [number, number, number]): boolean {
  const [ar, ak, ac] = a, [br, bk, bc] = b
  return (ar >= br && ak <= bk && ac <= bc) && (ar > br || ak < bk || ac < bc)
}

function _fastNonDominatedSort(objectives: [number, number, number][]): number[][] {
  const n = objectives.length
  const domCount = new Array(n).fill(0)
  const domSet: number[][] = Array.from({ length: n }, () => [])
  const ranks = new Array(n).fill(0)
  const fronts: number[][] = [[]]

  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      if (p === q) continue
      if (_dominates(objectives[p], objectives[q])) domSet[p].push(q)
      else if (_dominates(objectives[q], objectives[p])) domCount[p]++
    }
    if (domCount[p] === 0) { ranks[p] = 0; fronts[0].push(p) }
  }

  let i = 0
  while (fronts[i].length) {
    const next: number[] = []
    for (const p of fronts[i]) {
      for (const q of domSet[p]) {
        domCount[q]--
        if (domCount[q] === 0) { ranks[q] = i + 1; next.push(q) }
      }
    }
    i++
    fronts.push(next)
  }
  fronts.pop()
  return fronts
}

function _crowdingDistance(objectives: [number, number, number][], front: number[]): Record<number, number> {
  const dist: Record<number, number> = {}
  front.forEach(i => dist[i] = 0)
  const m = 3
  for (let o = 0; o < m; o++) {
    const sf = front.slice().sort((a, b) => objectives[a][o] - objectives[b][o])
    if (sf.length <= 2) { sf.forEach(i => dist[i] = Infinity); continue }
    dist[sf[0]] = Infinity; dist[sf[sf.length - 1]] = Infinity
    const mn = objectives[sf[0]][o], mx = objectives[sf[sf.length - 1]][o]
    const rng = mx !== mn ? mx - mn : 1
    for (let k = 1; k < sf.length - 1; k++) {
      dist[sf[k]] += (objectives[sf[k + 1]][o] - objectives[sf[k - 1]][o]) / rng
    }
  }
  return dist
}

function _tournament(pop: number[][], ranks: number[], crowd: Record<number, number>, size = 2): number[] {
  const n = pop.length
  const candidates: number[] = []
  while (candidates.length < size) {
    const c = Math.floor(Math.random() * n)
    if (!candidates.includes(c)) candidates.push(c)
  }
  let best = candidates[0]
  for (let k = 1; k < candidates.length; k++) {
    const c = candidates[k]
    if (ranks[c] < ranks[best] || (ranks[c] === ranks[best] && (crowd[c] ?? 0) > (crowd[best] ?? 0))) best = c
  }
  return pop[best].slice()
}

function _crossover(p1: number[], p2: number[]): number[] {
  const n = p1.length, c = new Array(n)
  for (let i = 0; i < n; i++) c[i] = Math.random() < 0.5 ? p1[i] : p2[i]
  return _normalizeWeights(c)
}

function _mutate(ind: number[], rate = 0.1): number[] {
  const n = ind.length, m = ind.slice()
  for (let i = 0; i < n; i++) {
    if (Math.random() < rate) m[i] += _mooNormalRandom() * 0.1
  }
  return _normalizeWeights(m)
}

function runMOOFrontend(config: MOOConfig): MOOResult {
  const rng = config.seed !== null ? new SeededRng(config.seed) : null
  const n = config.assets
  const er = (config.expected_returns?.length === n) ? config.expected_returns.slice() : _generateDefaultReturns(n, rng)
  const cov = (config.cov_matrix?.length === n && config.cov_matrix[0]?.length === n)
    ? config.cov_matrix.map(r => r.slice())
    : _generateDefaultCov(n, rng)

  const population: number[][] = []
  for (let i = 0; i < config.population_size; i++) population.push(_dirichletWeights(n, rng))

  for (let gen = 0; gen < config.generations; gen++) {
    const objectives: [number, number, number][] = population.map(w =>
      _evaluateWeights(w, er, cov, config.cost_per_sample, config.sample_size_base))

    const fronts = _fastNonDominatedSort(objectives)
    const ranks = new Array(population.length).fill(0)
    fronts.forEach((f, ri) => f.forEach(i => ranks[i] = ri))

    const crowd: Record<number, number> = {}
    fronts.forEach(f => Object.assign(crowd, _crowdingDistance(objectives, f)))

    const selected: number[][] = []
    let fi = 0
    while (fi < fronts.length && selected.length + fronts[fi].length <= config.population_size) {
      fronts[fi].forEach(i => selected.push(population[i]))
      fi++
    }
    if (selected.length < config.population_size && fi < fronts.length) {
      const rem = config.population_size - selected.length
      const fc = fronts[fi].map(i => [i, crowd[i] ?? 0] as [number, number])
      fc.sort((a, b) => b[1] - a[1])
      for (let k = 0; k < rem; k++) selected.push(population[fc[k][0]])
    }

    const offspring: number[][] = []
    for (let i = 0; i < config.population_size; i++) {
      const p1 = _tournament(selected, ranks, crowd)
      const p2 = _tournament(selected, ranks, crowd)
      offspring.push(_mutate(_crossover(p1, p2)))
    }

    const combined = selected.concat(offspring)
    const combObj: [number, number, number][] = combined.map(w =>
      _evaluateWeights(w, er, cov, config.cost_per_sample, config.sample_size_base))

    const combFronts = _fastNonDominatedSort(combObj)
    const combRanks = new Array(combined.length).fill(0)
    combFronts.forEach((f, ri) => f.forEach(i => combRanks[i] = ri))
    const combCrowd: Record<number, number> = {}
    combFronts.forEach(f => Object.assign(combCrowd, _crowdingDistance(combObj, f)))

    const nextPop: number[][] = []
    let ci = 0
    while (ci < combFronts.length && nextPop.length + combFronts[ci].length <= config.population_size) {
      combFronts[ci].forEach(i => nextPop.push(combined[i]))
      ci++
    }
    if (nextPop.length < config.population_size && ci < combFronts.length) {
      const rem = config.population_size - nextPop.length
      const fc = combFronts[ci].map(i => [i, combCrowd[i] ?? 0] as [number, number])
      fc.sort((a, b) => b[1] - a[1])
      for (let k = 0; k < rem; k++) nextPop.push(combined[fc[k][0]])
    }

    population.splice(0, population.length, ...nextPop)
  }

  const finalObj: [number, number, number][] = population.map(w =>
    _evaluateWeights(w, er, cov, config.cost_per_sample, config.sample_size_base))

  const finalFronts = _fastNonDominatedSort(finalObj)
  const finalRanks = new Array(population.length).fill(0)
  finalFronts.forEach((f, ri) => f.forEach(i => finalRanks[i] = ri))
  const finalCrowd: Record<number, number> = {}
  finalFronts.forEach(f => Object.assign(finalCrowd, _crowdingDistance(finalObj, f)))

  const rList = finalObj.map(o => o[0])
  const kList = finalObj.map(o => o[1])
  const cList = finalObj.map(o => o[2])

  const mn = (a: number[]) => Math.min(...a)
  const mx = (a: number[]) => Math.max(...a)
  const norm = (v: number, a: number[]) => mx(a) === mn(a) ? 0.5 : (v - mn(a)) / (mx(a) - mn(a))

  const scores = population.map((_, i) =>
    config.return_weight * norm(rList[i], rList) +
    config.risk_weight * (1 - norm(kList[i], kList)) +
    config.cost_weight * (1 - norm(cList[i], cList))
  )

  const all_solutions: MOOSolution[] = population.map((w, i) => ({
    weights: w.map(v => Math.round(v * 1e6) / 1e6),
    expected_return: Math.round(rList[i] * 1e6) / 1e6,
    risk: Math.round(kList[i] * 1e6) / 1e6,
    sample_cost: Math.round(cList[i] * 1e6) / 1e6,
    rank: finalRanks[i],
    crowding_distance: Math.round((finalCrowd[i] === Infinity ? 1e9 : finalCrowd[i] ?? 0) * 1e6) / 1e6,
    score: Math.round(scores[i] * 1e6) / 1e6
  }))

  const pareto = (finalFronts[0] || []).map(i => all_solutions[i])
  const bestIdx = scores.indexOf(Math.max(...scores))

  return {
    pareto_front: pareto,
    all_solutions,
    recommended: all_solutions[bestIdx],
    generations: config.generations,
    population_size: config.population_size
  }
}

export const useMCStore = defineStore('mc', () => {
  const currentScenario = ref<MCScenario>(SCENARIOS[0])
  const iterations = ref(1000)
  const result = ref<MCResult | null>(null)
  const testResult = ref<HypTestResult | null>(null)
  const isRunning = ref(false)

  const mooConfig = ref<MOOConfig>({
    assets: 5,
    population_size: 80,
    generations: 40,
    expected_returns: [],
    cov_matrix: [],
    cost_per_sample: 0.1,
    sample_size_base: 100,
    risk_weight: 0.4,
    return_weight: 0.4,
    cost_weight: 0.2,
    use_backend: false,
    seed: 42
  })
  const mooResult = ref<MOOResult | null>(null)
  const mooIsRunning = ref(false)
  const mooError = ref<string | null>(null)
  const mooAssetLabels = ref<string[]>([])

  function ensureMooAssets() {
    const n = mooConfig.value.assets
    if (mooConfig.value.expected_returns.length !== n) {
      mooConfig.value.expected_returns = new Array(n).fill(0).map((_, i) =>
        parseFloat((0.05 + 0.035 * i).toFixed(4)))
    }
    if (mooConfig.value.cov_matrix.length !== n || mooConfig.value.cov_matrix[0]?.length !== n) {
      const mat: number[][] = []
      for (let i = 0; i < n; i++) {
        mat[i] = []
        for (let j = 0; j < n; j++) {
          if (i === j) mat[i][j] = parseFloat((0.02 + 0.008 * i).toFixed(4))
          else mat[i][j] = parseFloat((0.003 * Math.min(i, j) / Math.max(i, j, 1)).toFixed(4))
        }
      }
      mooConfig.value.cov_matrix = mat
    }
    if (mooAssetLabels.value.length !== n) {
      mooAssetLabels.value = new Array(n).fill('').map((_, i) => `资产${String.fromCharCode(65 + i)}`)
    }
  }

  async function runMOO() {
    ensureMooAssets()
    mooIsRunning.value = true
    mooError.value = null
    try {
      if (mooConfig.value.use_backend) {
        const payload = {
          assets: mooConfig.value.assets,
          population_size: mooConfig.value.population_size,
          generations: mooConfig.value.generations,
          expected_returns: mooConfig.value.expected_returns,
          cov_matrix: mooConfig.value.cov_matrix,
          cost_per_sample: mooConfig.value.cost_per_sample,
          sample_size_base: mooConfig.value.sample_size_base,
          risk_weight: mooConfig.value.risk_weight,
          return_weight: mooConfig.value.return_weight,
          cost_weight: mooConfig.value.cost_weight,
          seed: mooConfig.value.seed
        }
        const res = await fetch('http://localhost:8000/api/multi-objective-optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
        if (!res.ok) throw new Error(`后端响应: ${res.status}`)
        mooResult.value = await res.json()
      } else {
        await new Promise(r => setTimeout(r, 30))
        mooResult.value = runMOOFrontend(mooConfig.value)
      }
    } catch (e: any) {
      mooError.value = e?.message || '优化失败，已回退到前端算法'
      try { mooResult.value = runMOOFrontend(mooConfig.value) } catch { /* noop */ }
    } finally {
      mooIsRunning.value = false
    }
  }

  function runSimulation() {
    isRunning.value = true
    setTimeout(() => { result.value = runMC(currentScenario.value, iterations.value); isRunning.value = false }, 10)
  }

  function runTest(g1: number[], g2: number[]) {
    const n1 = g1.length, n2 = g2.length
    const m1 = g1.reduce((a, b) => a + b, 0) / n1
    const m2 = g2.reduce((a, b) => a + b, 0) / n2
    const v1 = g1.reduce((s, x) => s + (x - m1) ** 2, 0) / (n1 - 1)
    const v2 = g2.reduce((s, x) => s + (x - m2) ** 2, 0) / (n2 - 1)
    const se = Math.sqrt(v1 / n1 + v2 / n2)
    const t = (m1 - m2) / se
    const df = Math.round((v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)))
    const pValue = 2 * (1 - Math.min(0.9999, Math.abs(t) / (Math.abs(t) + Math.sqrt(df))))
    testResult.value = { testType: 'Welch T检验', statistic: Math.round(t * 1000) / 1000, pValue: Math.round(pValue * 10000) / 10000, significant: pValue < 0.05, alpha: 0.05, df }
  }

  function setScenario(s: MCScenario) { currentScenario.value = s; result.value = null }

  const convergenceData = computed(() => {
    if (!result.value) return [] as [number, number][]
    return result.value.convergence.slice(0, 200).map((v, i): [number, number] => [i, Math.round(v * 100000) / 100000])
  })

  const histogramData = computed(() => {
    if (!result.value) return { xAxis: [] as number[], data: [] as number[] }
    const s = result.value.samples.slice(0, 1000)
    const mn = Math.min(...s), mx = Math.max(...s)
    const bins = 20, bs = (mx - mn) / bins || 1
    const counts = new Array(bins).fill(0)
    s.forEach(v => { counts[Math.min(bins - 1, Math.floor((v - mn) / bs))]++ })
    return { xAxis: Array.from({ length: bins }, (_, i) => Math.round((mn + i * bs) * 100) / 100), data: counts }
  })

  const mooScatterData = computed(() => {
    if (!mooResult.value) return [] as [number, number, number][]
    return mooResult.value.all_solutions.map(s => [s.risk, s.expected_return, s.sample_cost])
  })

  const mooParetoData = computed(() => {
    if (!mooResult.value) return [] as [number, number, number][]
    return mooResult.value.pareto_front.map(s => [s.risk, s.expected_return, s.sample_cost])
  })

  const mooRecommendedPoint = computed(() => {
    if (!mooResult.value) return null as [number, number, number] | null
    const r = mooResult.value.recommended
    return [r.risk, r.expected_return, r.sample_cost]
  })

  return {
    currentScenario, iterations, result, testResult, isRunning,
    convergenceData, histogramData, runSimulation, runTest, setScenario,
    mooConfig, mooResult, mooIsRunning, mooError, mooAssetLabels,
    ensureMooAssets, runMOO, mooScatterData, mooParetoData, mooRecommendedPoint
  }
})
