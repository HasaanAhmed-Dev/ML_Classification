import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { FEATURES } from './feature-schema';
import { SAMPLE_FLOWS } from './sample-flows';
import { PredictionResponse, PredictionService } from './prediction.service';

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly fields = FEATURES;
  readonly samples = SAMPLE_FLOWS;
  // On GitHub Pages there is no Flask server, so live prediction is disabled there.
  readonly deployedStatic = !['localhost', '127.0.0.1'].includes(window.location.hostname);

  values: Record<string, number | string | null> = {};
  readonly loadedSample = signal<number | null>(null);
  readonly loading = signal(false);
  readonly error = signal('');
  readonly result = signal<PredictionResponse | null>(null);
  readonly isDemo = signal(false);
  readonly bars = computed(() => {
    const r = this.result();
    if (!r) return [];
    return Object.entries(r.probabilities)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  });

  private api = inject(PredictionService);

  constructor() {
    this.clear();
  }

  loadSample(index: number) {
    const sample = this.samples[index];
    for (const f of this.fields) this.values[f.name] = sample.values[f.name];
    this.loadedSample.set(index);
    this.result.set(null);
    this.error.set('');
  }

  clear() {
    for (const f of this.fields) this.values[f.name] = null;
    this.loadedSample.set(null);
    this.result.set(null);
    this.error.set('');
  }

  private sampleUnchanged(): boolean {
    const index = this.loadedSample();
    if (index === null) return false;
    const sample = this.samples[index];
    return this.fields.every(f => Number(this.values[f.name]) === sample.values[f.name]);
  }

  submit() {
    this.error.set('');
    this.result.set(null);
    this.isDemo.set(false);

    if (this.deployedStatic) {
      const index = this.loadedSample();
      if (index !== null && this.sampleUnchanged()) {
        const s = this.samples[index];
        this.result.set({ prediction: s.prediction, confidence: s.confidence,
                          probabilities: s.probabilities });
        this.isDemo.set(true);
      } else {
        this.error.set('Static GitHub Pages demo: load an example flow unchanged to see its stored result. ' +
                       'Predictions for other values need Flask and Angular running locally.');
      }
      return;
    }

    const payload: Record<string, number> = {};
    for (const f of this.fields) {
      const raw = this.values[f.name];
      if (raw === null || raw === '' || raw === undefined) {
        this.error.set(`Enter a value for "${f.label}"`);
        return;
      }
      const numeric = Number(raw);
      if (!Number.isFinite(numeric)) {
        this.error.set(`Invalid number for "${f.label}"`);
        return;
      }
      payload[f.name] = numeric;
    }

    this.loading.set(true);
    this.api.predict(payload).subscribe({
      next: response => { this.result.set(response); this.loading.set(false); },
      error: err => {
        this.loading.set(false);
        this.error.set(err.status === 0
          ? 'Cannot reach the API. Is Flask running (Terminal 1) and Angular started with --proxy-config proxy.conf.json?'
          : `API error ${err.status}: ` + (err.error?.error || err.message || 'see the Flask terminal'));
      },
    });
  }
}
