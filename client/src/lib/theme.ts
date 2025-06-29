/**
 * Theme utilities for consistent color management across the app
 */

export const getThemeColors = (isDark: boolean) => {
  return {
    // Plotly chart colors
    plotly: {
      background: isDark ? '#0f172a' : '#ffffff',
      paper: isDark ? '#0f172a' : '#ffffff',
      text: isDark ? '#f8fafc' : '#0f172a',
      grid: isDark ? '#334155' : '#e2e8f0',
      axis: isDark ? '#f8fafc' : '#0f172a',
    },
    
    // Model architecture colors
    model: {
      embedding: isDark ? '#fbbf24' : '#f59e0b',
      attention: isDark ? '#f472b6' : '#ec4899',
      mlp: isDark ? '#34d399' : '#10b981',
      normalization: isDark ? '#a78bfa' : '#8b5cf6',
      output: isDark ? '#fb7185' : '#f43f5e',
      connection: isDark ? '#64748b' : '#475569',
      flow: isDark ? '#06b6d4' : '#0891b2',
    },
    
    // 3D visualization colors
    visualization: {
      primary: isDark ? '#3b82f6' : '#2563eb',
      secondary: isDark ? '#10b981' : '#059669',
      accent: isDark ? '#f59e0b' : '#d97706',
      highlight: isDark ? '#ef4444' : '#dc2626',
      muted: isDark ? '#6b7280' : '#9ca3af',
    },
    
    // Training metrics colors
    training: {
      loss: isDark ? '#ef4444' : '#dc2626',
      accuracy: isDark ? '#22c55e' : '#16a34a',
      learning_rate: isDark ? '#3b82f6' : '#2563eb',
      gradient: isDark ? '#a855f7' : '#9333ea',
      validation: isDark ? '#f59e0b' : '#d97706',
    }
  };
};

export const getPlotlyLayout = (isDark: boolean, title?: string) => {
  const colors = getThemeColors(isDark);
  
  return {
    paper_bgcolor: colors.plotly.background,
    plot_bgcolor: colors.plotly.background,
    font: {
      color: colors.plotly.text,
      family: 'Inter, system-ui, sans-serif',
      size: 12
    },
    title: title ? {
      text: title,
      font: {
        color: colors.plotly.text,
        size: 16,
        family: 'Inter, system-ui, sans-serif'
      },
      x: 0.5,
      xanchor: 'center'
    } : undefined,
    scene: {
      bgcolor: colors.plotly.background,
      xaxis: {
        gridcolor: colors.plotly.grid,
        color: colors.plotly.axis,
        showgrid: true,
        zeroline: false
      },
      yaxis: {
        gridcolor: colors.plotly.grid,
        color: colors.plotly.axis,
        showgrid: true,
        zeroline: false
      },
      zaxis: {
        gridcolor: colors.plotly.grid,
        color: colors.plotly.axis,
        showgrid: true,
        zeroline: false
      }
    },
    xaxis: {
      gridcolor: colors.plotly.grid,
      color: colors.plotly.axis,
      showgrid: true,
      zeroline: false
    },
    yaxis: {
      gridcolor: colors.plotly.grid,
      color: colors.plotly.axis,
      showgrid: true,
      zeroline: false
    }
  };
};

export const getModelArchitectureColors = (isDark: boolean) => {
  const colors = getThemeColors(isDark);
  return colors.model;
};

export const getVisualizationColors = (isDark: boolean) => {
  const colors = getThemeColors(isDark);
  return colors.visualization;
};

export const getTrainingColors = (isDark: boolean) => {
  const colors = getThemeColors(isDark);
  return colors.training;
};