<template>
  <div class="min-h-screen bg-slate-900 text-slate-200">
    <header class="border-b border-slate-700 px-6 py-4">
      <h1 class="text-2xl font-bold text-cyan-400">蒙特卡洛模拟与统计假设检验平台</h1>
      <p class="text-sm text-slate-500 mt-1">随机采样模拟 · 6种MC场景 · 假设检验 · 置信区间可视化 · 多目标优化</p>
      <div class="mt-3 flex gap-2">
        <button v-for="t in TABS" :key="t.id" @click="activeTab = t.id"
          :class="['px-4 py-1.5 rounded text-sm font-medium transition-all', activeTab === t.id ? 'bg-cyan-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700']">
          {{ t.name }}
        </button>
      </div>
    </header>

    <div v-show="activeTab === 'mc'" class="flex flex-col lg:flex-row gap-4 p-4">
      <div class="lg:w-1/4 space-y-4">
        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <h3 class="text-sm font-bold text-slate-400 mb-3">模拟场景</h3>
          <div class="space-y-1">
            <div v-for="s in SCENARIOS" :key="s.id" @click="store.setScenario(s)"
              :class="['cursor-pointer p-2 rounded border text-sm transition-all', store.currentScenario.id === s.id ? 'border-cyan-500 bg-cyan-900/30 text-cyan-400' : 'border-slate-700 text-slate-300 hover:border-slate-500']">
              <div class="font-bold">{{ s.name }}</div>
              <div class="text-xs text-slate-500 mt-0.5">{{ s.description }}</div>
            </div>
          </div>
        </div>
        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <h3 class="text-sm font-bold text-slate-400 mb-3">参数控制</h3>
          <label class="text-xs text-slate-500">迭代次数: {{ store.iterations }}</label>
          <input type="range" min="100" max="5000" step="100" v-model.number="store.iterations" class="w-full mt-1 mb-3 accent-cyan-500" />
          <button @click="store.runSimulation" :disabled="store.isRunning" class="w-full py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 rounded text-sm font-bold">
            {{ store.isRunning ? '运行中...' : '▶ 开始模拟' }}
          </button>
        </div>
        <div v-if="store.result" class="bg-slate-800 rounded-lg p-4 border border-slate-700 text-sm">
          <h3 class="text-sm font-bold text-slate-400 mb-3">模拟结果</h3>
          <div class="space-y-2">
            <div class="flex justify-between"><span class="text-slate-500">估算值</span><span class="text-cyan-400 font-bold font-mono">{{ store.result.estimate.toFixed(6) }}</span></div>
            <div v-if="store.result.trueValue !== undefined" class="flex justify-between"><span class="text-slate-500">真实值</span><span class="text-green-400 font-mono">{{ store.result.trueValue.toFixed(6) }}</span></div>
            <div v-if="store.result.error !== undefined" class="flex justify-between"><span class="text-slate-500">误差</span><span class="text-orange-400 font-mono">{{ store.result.error.toFixed(6) }}</span></div>
            <div class="flex justify-between"><span class="text-slate-500">样本数</span><span class="text-slate-300">{{ store.result.iterations }}</span></div>
          </div>
        </div>
      </div>
      <div class="lg:w-3/4 space-y-4">
        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <h3 class="text-sm font-bold text-slate-400 mb-3">收敛过程</h3>
          <div ref="convergenceRef" class="w-full rounded" style="height:240px;background:#0f172a;"></div>
        </div>
        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <h3 class="text-sm font-bold text-slate-400 mb-3">样本分布直方图</h3>
          <div ref="histogramRef" class="w-full rounded" style="height:220px;background:#0f172a;"></div>
        </div>
        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <h3 class="text-sm font-bold text-slate-400 mb-3">假设检验 (独立样本 T 检验)</h3>
          <div class="grid grid-cols-2 gap-4 mb-3">
            <div>
              <label class="text-xs text-slate-500">样本组A (逗号分隔)</label>
              <textarea v-model="group1Input" rows="2" class="w-full mt-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-cyan-500 resize-none"></textarea>
            </div>
            <div>
              <label class="text-xs text-slate-500">样本组B (逗号分隔)</label>
              <textarea v-model="group2Input" rows="2" class="w-full mt-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-cyan-500 resize-none"></textarea>
            </div>
          </div>
          <button @click="runTest" class="px-4 py-1.5 bg-purple-600 hover:bg-purple-500 rounded text-sm">执行T检验</button>
          <div v-if="store.testResult" class="mt-3 grid grid-cols-4 gap-3 text-sm">
            <div class="bg-slate-900 rounded p-2 text-center"><div class="text-xs text-slate-500 mb-1">统计量 t</div><div class="text-cyan-400 font-bold font-mono">{{ store.testResult.statistic }}</div></div>
            <div class="bg-slate-900 rounded p-2 text-center"><div class="text-xs text-slate-500 mb-1">p 值</div><div class="font-bold font-mono" :class="store.testResult.significant ? 'text-red-400' : 'text-green-400'">{{ store.testResult.pValue }}</div></div>
            <div class="bg-slate-900 rounded p-2 text-center"><div class="text-xs text-slate-500 mb-1">自由度 df</div><div class="text-slate-300 font-mono">{{ store.testResult.df }}</div></div>
            <div class="bg-slate-900 rounded p-2 text-center"><div class="text-xs text-slate-500 mb-1">显著性</div><div class="text-xs font-bold" :class="store.testResult.significant ? 'text-red-400' : 'text-green-400'">{{ store.testResult.significant ? '显著(p<0.05)' : '不显著' }}</div></div>
          </div>
        </div>
      </div>
    </div>

    <div v-show="activeTab === 'moo'" class="flex flex-col lg:flex-row gap-4 p-4">
      <div class="lg:w-1/3 space-y-4">
        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <h3 class="text-sm font-bold text-emerald-400 mb-3">🎯 多目标优化配置</h3>
          <div class="space-y-3 text-sm">
            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="text-xs text-slate-500">资产数量: {{ store.mooConfig.assets }}</label>
                <input type="range" min="3" max="10" step="1" v-model.number="store.mooConfig.assets" @change="store.ensureMooAssets" class="w-full mt-1 accent-emerald-500" />
              </div>
              <div>
                <label class="text-xs text-slate-500">种群大小: {{ store.mooConfig.population_size }}</label>
                <input type="range" min="40" max="200" step="10" v-model.number="store.mooConfig.population_size" class="w-full mt-1 accent-emerald-500" />
              </div>
            </div>
            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="text-xs text-slate-500">进化代数: {{ store.mooConfig.generations }}</label>
                <input type="range" min="10" max="100" step="5" v-model.number="store.mooConfig.generations" class="w-full mt-1 accent-emerald-500" />
              </div>
              <div>
                <label class="text-xs text-slate-500">样本成本基数: {{ store.mooConfig.sample_size_base }}</label>
                <input type="range" min="50" max="500" step="10" v-model.number="store.mooConfig.sample_size_base" class="w-full mt-1 accent-emerald-500" />
              </div>
            </div>
            <div>
              <label class="text-xs text-slate-500">单位样本成本: {{ store.mooConfig.cost_per_sample.toFixed(2) }}</label>
              <input type="range" min="0.01" max="0.5" step="0.01" v-model.number="store.mooConfig.cost_per_sample" class="w-full mt-1 accent-emerald-500" />
            </div>
            <div class="pt-2 border-t border-slate-700">
              <div class="text-xs font-bold text-slate-400 mb-2">目标偏好权重 (总和=1)</div>
              <div class="space-y-2">
                <div>
                  <div class="flex justify-between text-xs mb-0.5"><span class="text-cyan-400">收益最大化</span><span class="font-mono text-slate-300">{{ (store.mooConfig.return_weight*100).toFixed(0) }}%</span></div>
                  <input type="range" min="0" max="1" step="0.05" v-model.number="store.mooConfig.return_weight" @input="normalWeights('return')" class="w-full accent-cyan-500" />
                </div>
                <div>
                  <div class="flex justify-between text-xs mb-0.5"><span class="text-rose-400">风险最小化</span><span class="font-mono text-slate-300">{{ (store.mooConfig.risk_weight*100).toFixed(0) }}%</span></div>
                  <input type="range" min="0" max="1" step="0.05" v-model.number="store.mooConfig.risk_weight" @input="normalWeights('risk')" class="w-full accent-rose-500" />
                </div>
                <div>
                  <div class="flex justify-between text-xs mb-0.5"><span class="text-amber-400">成本最小化</span><span class="font-mono text-slate-300">{{ (store.mooConfig.cost_weight*100).toFixed(0) }}%</span></div>
                  <input type="range" min="0" max="1" step="0.05" v-model.number="store.mooConfig.cost_weight" @input="normalWeights('cost')" class="w-full accent-amber-500" />
                </div>
              </div>
            </div>
            <div class="flex gap-2 pt-2">
              <label class="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer">
                <input type="checkbox" v-model="store.mooConfig.use_backend" class="accent-emerald-500" />
                使用后端API
              </label>
              <div class="flex-1"></div>
              <label class="flex items-center gap-1.5 text-xs text-slate-400">
                随机种子
                <input type="number" v-model.number="store.mooConfig.seed" :disabled="store.mooConfig.seed===null" class="w-16 bg-slate-900 border border-slate-600 rounded px-1.5 py-0.5 text-xs focus:outline-none focus:border-emerald-500" />
              </label>
            </div>
            <button @click="store.runMOO" :disabled="store.mooIsRunning" class="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded text-sm font-bold mt-2">
              {{ store.mooIsRunning ? '⏳ NSGA-II 进化中...' : '🚀 开始多目标优化' }}
            </button>
            <div v-if="store.mooError" class="text-xs text-orange-400 bg-orange-900/20 rounded p-2 border border-orange-700/50">
              ⚠ {{ store.mooError }}
            </div>
          </div>
        </div>

        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <h3 class="text-sm font-bold text-slate-400 mb-3">📊 资产预期收益率</h3>
          <div class="grid grid-cols-1 gap-2" v-if="store.mooConfig.expected_returns.length">
            <div v-for="(r, i) in store.mooConfig.expected_returns" :key="'er'+i" class="flex items-center gap-2">
              <span class="text-xs text-slate-500 w-12">{{ store.mooAssetLabels[i] || ('A'+i) }}</span>
              <input type="number" step="0.005" v-model.number="store.mooConfig.expected_returns[i]" class="flex-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-emerald-500" />
              <span class="text-xs text-slate-500 w-8">→{{ (store.mooConfig.expected_returns[i]*100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>

        <div v-if="store.mooResult" class="bg-slate-800 rounded-lg p-4 border-2 border-emerald-600/50">
          <h3 class="text-sm font-bold text-emerald-400 mb-3">⭐ 推荐方案 (加权最优)</h3>
          <div class="grid grid-cols-3 gap-2 text-sm mb-3">
            <div class="bg-slate-900 rounded p-2 text-center"><div class="text-xs text-slate-500 mb-1">综合评分</div><div class="text-emerald-400 font-bold font-mono">{{ store.mooResult.recommended.score?.toFixed(4) }}</div></div>
            <div class="bg-slate-900 rounded p-2 text-center"><div class="text-xs text-slate-500 mb-1">预期收益</div><div class="text-cyan-400 font-bold font-mono">{{ (store.mooResult.recommended.expected_return*100).toFixed(2) }}%</div></div>
            <div class="bg-slate-900 rounded p-2 text-center"><div class="text-xs text-slate-500 mb-1">风险σ</div><div class="text-rose-400 font-bold font-mono">{{ (store.mooResult.recommended.risk*100).toFixed(2) }}%</div></div>
          </div>
          <div class="bg-slate-900 rounded p-2 text-center mb-3"><div class="text-xs text-slate-500 mb-1">样本成本</div><div class="text-amber-400 font-bold font-mono">{{ store.mooResult.recommended.sample_cost.toFixed(2) }}</div></div>
          <div>
            <div class="text-xs text-slate-500 mb-2">资产配置权重</div>
            <div class="space-y-1.5">
              <div v-for="(w, i) in store.mooResult.recommended.weights" :key="'w'+i" class="flex items-center gap-2">
                <span class="text-xs text-slate-400 w-12">{{ store.mooAssetLabels[i] || ('A'+i) }}</span>
                <div class="flex-1 bg-slate-900 rounded-full h-3 overflow-hidden">
                  <div class="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full transition-all" :style="{ width: (w*100).toFixed(1)+'%' }"></div>
                </div>
                <span class="text-xs font-mono text-slate-300 w-14 text-right">{{ (w*100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="lg:w-2/3 space-y-4">
        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <h3 class="text-sm font-bold text-slate-400 mb-3">📈 Pareto 前沿 (风险-收益空间, 颜色=样本成本)</h3>
          <div ref="paretoRef" class="w-full rounded" style="height:360px;background:#0f172a;"></div>
        </div>
        <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <h3 class="text-sm font-bold text-slate-400 mb-3">🥧 推荐方案资产权重分布</h3>
            <div ref="pieRef" class="w-full rounded" style="height:280px;background:#0f172a;"></div>
          </div>
          <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <h3 class="text-sm font-bold text-slate-400 mb-3">🎯 三目标权衡雷达</h3>
            <div ref="radarRef" class="w-full rounded" style="height:280px;background:#0f172a;"></div>
          </div>
        </div>
        <div class="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <div class="flex justify-between items-center mb-3">
            <h3 class="text-sm font-bold text-slate-400">📋 Pareto 前沿解集 ({{ store.mooResult?.pareto_front.length || 0 }} 个非支配解)</h3>
            <div class="text-xs text-slate-500">种群: {{ store.mooResult?.population_size || 0 }} | 代数: {{ store.mooResult?.generations || 0 }}</div>
          </div>
          <div v-if="store.mooResult?.pareto_front.length" class="overflow-x-auto max-h-64 overflow-y-auto">
            <table class="w-full text-xs">
              <thead class="sticky top-0 bg-slate-800">
                <tr class="text-slate-500 border-b border-slate-700">
                  <th class="text-left py-2 px-2">#</th>
                  <th class="text-right py-2 px-2">预期收益↑</th>
                  <th class="text-right py-2 px-2">风险σ↓</th>
                  <th class="text-right py-2 px-2">样本成本↓</th>
                  <th class="text-right py-2 px-2">综合得分</th>
                  <th v-for="(lb, i) in store.mooAssetLabels.slice(0, 6)" :key="'th'+i" class="text-right py-2 px-1 font-mono">{{ lb }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(s, idx) in sortedPareto" :key="idx" class="border-b border-slate-800 hover:bg-slate-900/50" :class="isRecommended(s) ? 'bg-emerald-900/20' : ''">
                  <td class="py-1.5 px-2 text-slate-400">{{ isRecommended(s) ? '⭐' : idx+1 }}</td>
                  <td class="py-1.5 px-2 text-right font-mono text-cyan-400">{{ (s.expected_return*100).toFixed(2) }}%</td>
                  <td class="py-1.5 px-2 text-right font-mono text-rose-400">{{ (s.risk*100).toFixed(2) }}%</td>
                  <td class="py-1.5 px-2 text-right font-mono text-amber-400">{{ s.sample_cost.toFixed(2) }}</td>
                  <td class="py-1.5 px-2 text-right font-mono text-emerald-400 font-bold">{{ s.score?.toFixed(3) }}</td>
                  <td v-for="(w, i) in s.weights.slice(0, 6)" :key="'td'+i" class="py-1.5 px-1 text-right font-mono text-slate-300">{{ (w*100).toFixed(0) }}%</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else class="text-center text-slate-600 py-10 text-sm">运行优化后显示 Pareto 解集</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, computed } from 'vue'
import * as echarts from 'echarts'
import { useMCStore, SCENARIOS, type MOOSolution } from './store/mc'

const TABS = [
  { id: 'mc', name: '🎲 蒙特卡洛模拟' },
  { id: 'moo', name: '🎯 多目标优化' }
]

const activeTab = ref('mc')
const store = useMCStore()
const convergenceRef = ref<HTMLDivElement | null>(null)
const histogramRef = ref<HTMLDivElement | null>(null)
const paretoRef = ref<HTMLDivElement | null>(null)
const pieRef = ref<HTMLDivElement | null>(null)
const radarRef = ref<HTMLDivElement | null>(null)
const group1Input = ref('5.1,4.8,5.3,4.9,5.2,5.0,4.7,5.1,5.4,4.8')
const group2Input = ref('4.6,4.2,4.9,4.3,4.5,4.7,4.4,4.8,4.1,4.6')
let convChart: echarts.ECharts | null = null
let histChart: echarts.ECharts | null = null
let paretoChart: echarts.ECharts | null = null
let pieChart: echarts.ECharts | null = null
let radarChart: echarts.ECharts | null = null

function normalWeights(changed: 'return' | 'risk' | 'cost') {
  const c = store.mooConfig
  let r = c.return_weight, k = c.risk_weight, x = c.cost_weight
  const total = r + k + x
  if (total > 0) { r /= total; k /= total; x /= total }
  else { r = k = x = 1/3 }
  c.return_weight = Math.round(r * 20) / 20
  c.risk_weight = Math.round(k * 20) / 20
  c.cost_weight = Math.max(0, 1 - c.return_weight - c.risk_weight)
}

const sortedPareto = computed(() => {
  if (!store.mooResult) return [] as MOOSolution[]
  return store.mooResult.pareto_front.slice().sort((a, b) => (b.score || 0) - (a.score || 0))
})

function isRecommended(s: MOOSolution): boolean {
  if (!store.mooResult) return false
  const r = store.mooResult.recommended
  return Math.abs(s.expected_return - r.expected_return) < 1e-8 &&
         Math.abs(s.risk - r.risk) < 1e-8 &&
         Math.abs(s.sample_cost - r.sample_cost) < 1e-8
}

function initCharts() {
  if (convergenceRef.value) convChart = echarts.init(convergenceRef.value, 'dark')
  if (histogramRef.value) histChart = echarts.init(histogramRef.value, 'dark')
  if (paretoRef.value) paretoChart = echarts.init(paretoRef.value, 'dark')
  if (pieRef.value) pieChart = echarts.init(pieRef.value, 'dark')
  if (radarRef.value) radarChart = echarts.init(radarRef.value, 'dark')
}

function updateCharts() {
  if (convChart && store.convergenceData.length > 0) {
    convChart.setOption({
      backgroundColor: '#0f172a',
      grid: { top: 20, bottom: 35, left: 65, right: 20 },
      xAxis: { type: 'value', axisLabel: { color: '#94a3b8', fontSize: 10 } },
      yAxis: { type: 'value', axisLabel: { color: '#94a3b8', fontSize: 10 } },
      series: [{ type: 'line', data: store.convergenceData, smooth: true, lineStyle: { color: '#06b6d4', width: 2 }, areaStyle: { color: 'rgba(6,182,212,0.1)' }, symbol: 'none' }],
      tooltip: { trigger: 'axis', backgroundColor: '#1e293b', borderColor: '#475569' }
    })
  }
  if (histChart && store.histogramData.xAxis.length > 0) {
    histChart.setOption({
      backgroundColor: '#0f172a',
      grid: { top: 15, bottom: 40, left: 55, right: 15 },
      xAxis: { type: 'category', data: store.histogramData.xAxis, axisLabel: { color: '#94a3b8', fontSize: 9, rotate: 30 } },
      yAxis: { type: 'value', axisLabel: { color: '#94a3b8', fontSize: 10 } },
      series: [{ type: 'bar', data: store.histogramData.data, itemStyle: { color: '#8b5cf6' } }],
      tooltip: { trigger: 'axis', backgroundColor: '#1e293b', borderColor: '#475569' }
    })
  }
}

function updateMooCharts() {
  if (!paretoChart) return
  const all = store.mooScatterData
  const pareto = store.mooParetoData
  const rec = store.mooRecommendedPoint

  if (all.length === 0) {
    paretoChart.setOption({
      backgroundColor: '#0f172a',
      title: { text: '运行优化以查看 Pareto 前沿', left: 'center', top: 'center', textStyle: { color: '#475569', fontSize: 14 } },
      xAxis: {}, yAxis: {}, series: []
    })
  } else {
    const risks = all.map(p => p[0]), returns = all.map(p => p[1]), costs = all.map(p => p[2])
    const minC = Math.min(...costs), maxC = Math.max(...costs)
    paretoChart.setOption({
      backgroundColor: '#0f172a',
      grid: { top: 25, bottom: 55, left: 70, right: 70 },
      tooltip: {
        backgroundColor: '#1e293b', borderColor: '#475569',
        formatter: (p: any) => {
          const d = p.data
          return `${p.seriesName}<br/>风险σ: ${(d[0]*100).toFixed(2)}%<br/>收益: ${(d[1]*100).toFixed(2)}%<br/>成本: ${d[2].toFixed(2)}`
        }
      },
      xAxis: {
        type: 'value', name: '风险 (σ)', nameLocation: 'middle', nameGap: 30,
        axisLabel: { color: '#94a3b8', fontSize: 10, formatter: (v: number) => (v*100).toFixed(0)+'%' },
        nameTextStyle: { color: '#94a3b8' }
      },
      yAxis: {
        type: 'value', name: '预期收益', nameLocation: 'middle', nameGap: 50,
        axisLabel: { color: '#94a3b8', fontSize: 10, formatter: (v: number) => (v*100).toFixed(0)+'%' },
        nameTextStyle: { color: '#94a3b8' }
      },
      visualMap: {
        show: true, min: minC, max: maxC, dimension: 2, orient: 'vertical', right: 10, top: 'center',
        text: ['成本↑', '成本↓'], inRange: { color: ['#10b981', '#f59e0b', '#ef4444'] },
        textStyle: { color: '#94a3b8', fontSize: 10 }
      },
      series: [
        {
          name: '所有解', type: 'scatter', data: all,
          symbolSize: 8, itemStyle: { opacity: 0.45, borderWidth: 0 },
          emphasis: { itemStyle: { borderColor: '#fff', borderWidth: 1, opacity: 1 } }
        },
        {
          name: 'Pareto前沿', type: 'scatter', data: pareto,
          symbolSize: 14, itemStyle: { opacity: 0.95, borderColor: '#22d3ee', borderWidth: 2 },
          encode: { x: 0, y: 1, tooltip: [0,1,2] }
        },
        rec ? {
          name: '★推荐方案', type: 'scatter', data: [rec],
          symbol: 'diamond', symbolSize: 22,
          itemStyle: { color: '#f472b6', borderColor: '#fbcfe8', borderWidth: 3, shadowBlur: 15, shadowColor: '#f472b6' },
          encode: { x: 0, y: 1 }
        } : { name: '推荐', type: 'scatter', data: [] }
      ]
    })
  }

  if (pieChart) {
    if (!store.mooResult) {
      pieChart.setOption({ backgroundColor: '#0f172a', title: { text: '运行优化后显示', left: 'center', top: 'center', textStyle: { color: '#475569', fontSize: 13 } }, series: [] })
    } else {
      pieChart.setOption({
        backgroundColor: '#0f172a',
        tooltip: { trigger: 'item', backgroundColor: '#1e293b', borderColor: '#475569', formatter: '{b}: {d}%' },
        legend: { orient: 'vertical', right: 10, top: 'center', textStyle: { color: '#94a3b8', fontSize: 11 } },
        color: ['#10b981', '#06b6d4', '#8b5cf6', '#f59e0b', '#ef4444', '#ec4899', '#6366f1', '#14b8a6', '#f97316', '#3b82f6'],
        series: [{
          type: 'pie', radius: ['40%', '72%'], center: ['40%', '50%'], avoidLabelOverlap: true,
          itemStyle: { borderRadius: 4, borderColor: '#0f172a', borderWidth: 2 },
          label: { show: false },
          data: store.mooResult.recommended.weights.map((w, i) => ({
            name: store.mooAssetLabels[i] || ('A'+i), value: Math.round(w * 1000) / 10
          }))
        }]
      })
    }
  }

  if (radarChart) {
    if (!store.mooResult || !store.mooResult.all_solutions.length) {
      radarChart.setOption({ backgroundColor: '#0f172a', title: { text: '运行优化后显示', left: 'center', top: 'center', textStyle: { color: '#475569', fontSize: 13 } }, series: [] })
    } else {
      const all = store.mooResult.all_solutions
      const maxR = Math.max(...all.map(s => s.expected_return))
      const maxK = Math.max(...all.map(s => s.risk))
      const maxC = Math.max(...all.map(s => s.sample_cost))
      const r = store.mooResult.recommended
      const bestR = Math.max(...all.map(s => s.expected_return))
      const bestK = Math.min(...all.map(s => s.risk))
      const bestC = Math.min(...all.map(s => s.sample_cost))
      radarChart.setOption({
        backgroundColor: '#0f172a',
        tooltip: { backgroundColor: '#1e293b', borderColor: '#475569' },
        legend: { top: 5, textStyle: { color: '#94a3b8', fontSize: 11 }, data: ['推荐方案', 'Pareto中位数', '理论最优'] },
        radar: {
          center: ['50%', '58%'], radius: '65%',
          indicator: [
            { name: '收益 (↑)', max: 100, color: '#94a3b8' },
            { name: '低风险 (↑)', max: 100, color: '#94a3b8' },
            { name: '低成本 (↑)', max: 100, color: '#94a3b8' }
          ],
          axisName: { color: '#cbd5e1', fontSize: 12, fontWeight: 'bold' },
          splitArea: { areaStyle: { color: ['rgba(30,41,59,0.6)', 'rgba(15,23,42,0.6)'] } },
          axisLine: { lineStyle: { color: '#475569' } },
          splitLine: { lineStyle: { color: '#334155' } }
        },
        series: [{
          type: 'radar',
          data: [
            {
              value: [
                Math.round(r.expected_return / maxR * 100),
                Math.round((1 - r.risk / maxK) * 100),
                Math.round((1 - r.sample_cost / maxC) * 100)
              ],
              name: '推荐方案',
              lineStyle: { color: '#10b981', width: 3 },
              areaStyle: { color: 'rgba(16,185,129,0.3)' },
              itemStyle: { color: '#10b981' }
            },
            {
              value: [
                Math.round(50),
                Math.round(50),
                Math.round(50)
              ],
              name: 'Pareto中位数',
              lineStyle: { color: '#8b5cf6', type: 'dashed' },
              itemStyle: { color: '#8b5cf6' }
            },
            {
              value: [
                Math.round(bestR / maxR * 100),
                Math.round((1 - bestK / maxK) * 100),
                Math.round((1 - bestC / maxC) * 100)
              ],
              name: '理论最优(乌托邦)',
              lineStyle: { color: '#f472b6', type: 'dotted' },
              areaStyle: { color: 'rgba(244,114,182,0.1)' },
              itemStyle: { color: '#f472b6' }
            }
          ]
        }]
      })
    }
  }
}

function runTest() {
  const g1 = group1Input.value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
  const g2 = group2Input.value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
  if (g1.length > 1 && g2.length > 1) store.runTest(g1, g2)
}

onMounted(() => {
  initCharts()
  store.runSimulation()
  store.ensureMooAssets()
  setTimeout(() => { store.runMOO() }, 200)
})

watch(() => store.result, () => updateCharts(), { deep: true })
watch(() => [store.mooResult, activeTab.value], () => { if (activeTab.value === 'moo') setTimeout(updateMooCharts, 30) }, { deep: true })

window.addEventListener('resize', () => {
  ;[convChart, histChart, paretoChart, pieChart, radarChart].forEach(c => c && c.resize())
})
</script>
