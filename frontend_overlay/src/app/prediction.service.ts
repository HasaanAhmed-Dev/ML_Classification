import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  task?: string;
  probabilities: Record<string, number>;
  model_version?: string;
  disclaimer?: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private http = inject(HttpClient);

  // Relative URL: in local development the Angular proxy forwards /api/predict to Flask /predict.
  predict(payload: Record<string, number>): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>('/api/predict', payload);
  }
}
