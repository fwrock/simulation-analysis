#!/usr/bin/env python3
"""
Análise de Escalabilidade dos Simuladores
Investiga as discrepâncias no número de eventos conforme aumenta o número de veículos
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class ScalabilityAnalyzer:
    def __init__(self, output_dir="/home/dean/PhD/scripts/simulations_analysis/output"):
        self.output_dir = Path(output_dir)
        self.scenarios = [1000, 2500, 5000, 7500]
        self.data = {}
        
    def load_comparison_data(self):
        """Carrega dados de comparação de todos os cenários"""
        
        scenario_files = {
            1000: "1000/comparison_bfs_cenario_1000_viagens_2_vs_events.json",
            2500: "2500/comparison_bfs_cenario_2500_viagens_3_vs_events.json", 
            5000: "5000/comparison_bfs_cenario_5000_viagens_1_vs_events.json",
            7500: "7500/comparison_bfs_cenario_7500_viagens_1_vs_events.json"
        }
        
        for vehicles, file_path in scenario_files.items():
            full_path = self.output_dir / file_path
            
            if full_path.exists():
                with open(full_path, 'r') as f:
                    data = json.load(f)
                    self.data[vehicles] = data
                    print(f"✅ Carregado cenário {vehicles} veículos")
            else:
                print(f"❌ Arquivo não encontrado: {full_path}")
    
    def extract_event_metrics(self):
        """Extrai métricas de eventos de cada cenário"""
        
        metrics = []
        
        for vehicles, data in self.data.items():
            htc_summary = data.get('htc_summary', {})
            ref_summary = data.get('ref_summary', {})
            
            # Eventos por tipo HTC
            htc_events = htc_summary.get('events_by_type', {})
            htc_total = htc_summary.get('total_events', 0)
            htc_enter = htc_events.get('enter_link', 0)
            htc_leave = htc_events.get('leave_link', 0)
            
            # Eventos por tipo Interscsimulator
            ref_events = ref_summary.get('events_by_type', {})
            ref_total = ref_summary.get('total_events', 0)
            ref_enter = ref_events.get('enter_link', 0)
            ref_leave = ref_events.get('leave_link', 0)
            
            metrics.append({
                'vehicles': vehicles,
                'htc_total_events': htc_total,
                'ref_total_events': ref_total,
                'htc_enter_events': htc_enter,
                'htc_leave_events': htc_leave,
                'ref_enter_events': ref_enter,
                'ref_leave_events': ref_leave,
                'total_diff': ref_total - htc_total,
                'enter_leave_diff': (ref_enter + ref_leave) - (htc_enter + htc_leave),
                'events_per_vehicle_htc': htc_total / vehicles if vehicles > 0 else 0,
                'events_per_vehicle_ref': ref_total / vehicles if vehicles > 0 else 0
            })
        
        return pd.DataFrame(metrics)
    
    def analyze_scalability_issues(self, df):
        """Analisa problemas de escalabilidade"""
        
        print("\n" + "="*80)
        print("🔍 ANÁLISE DE ESCALABILIDADE DOS SIMULADORES")
        print("="*80)
        
        # Análise da diferença absoluta
        print("\n📊 DIFERENÇAS ABSOLUTAS DE EVENTOS:")
        print("-" * 50)
        for _, row in df.iterrows():
            vehicles = int(row['vehicles'])
            diff = int(row['total_diff'])
            enter_leave_diff = int(row['enter_leave_diff'])
            
            print(f"🚗 {vehicles:,} veículos:")
            print(f"   📈 Total: {diff:,} eventos a mais no Interscsimulator")
            print(f"   🔗 Enter/Leave: {enter_leave_diff:,} eventos a mais")
            print()
        
        # Análise de crescimento relativo
        print("\n📈 ANÁLISE DE CRESCIMENTO RELATIVO:")
        print("-" * 50)
        
        base_1000 = df[df['vehicles'] == 1000].iloc[0]
        base_diff = base_1000['total_diff']
        
        print(f"📌 Baseline (1000 veículos): {int(base_diff):,} eventos de diferença")
        print()
        
        for _, row in df.iterrows():
            vehicles = int(row['vehicles'])
            if vehicles == 1000:
                continue
                
            current_diff = row['total_diff']
            growth_factor = current_diff / base_diff if base_diff != 0 else 0
            expected_diff = base_diff * (vehicles / 1000)  # Crescimento linear esperado
            excess = current_diff - expected_diff
            
            print(f"🚗 {vehicles:,} veículos:")
            print(f"   📊 Diferença atual: {int(current_diff):,} eventos")
            print(f"   📏 Crescimento esperado linear: {int(expected_diff):,} eventos")
            print(f"   ⚠️  Excesso: {int(excess):,} eventos ({growth_factor:.1f}x do baseline)")
            print(f"   🔍 Fator de crescimento: {growth_factor:.2f}x")
            print()
        
        # Análise de eventos por veículo
        print("\n🚗 EVENTOS POR VEÍCULO:")
        print("-" * 50)
        for _, row in df.iterrows():
            vehicles = int(row['vehicles'])
            htc_per_vehicle = row['events_per_vehicle_htc']
            ref_per_vehicle = row['events_per_vehicle_ref']
            
            print(f"🚗 {vehicles:,} veículos:")
            print(f"   📊 HTC: {htc_per_vehicle:.1f} eventos/veículo")
            print(f"   📊 Interscsimulator: {ref_per_vehicle:.1f} eventos/veículo")
            print(f"   📈 Diferença: {ref_per_vehicle - htc_per_vehicle:.1f} eventos/veículo")
            print()
    
    def create_scalability_plots(self, df):
        """Cria gráficos de análise de escalabilidade"""
        
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise de Escalabilidade dos Simuladores', fontsize=16, fontweight='bold')
        
        # 1. Total de eventos por número de veículos
        ax1 = axes[0, 0]
        ax1.plot(df['vehicles'], df['htc_total_events'], 'o-', label='HTC', linewidth=2, markersize=8)
        ax1.plot(df['vehicles'], df['ref_total_events'], 's-', label='Interscsimulator', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Veículos')
        ax1.set_ylabel('Total de Eventos')
        ax1.set_title('Total de Eventos vs Número de Veículos')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Adicionar valores nos pontos
        for _, row in df.iterrows():
            ax1.annotate(f"{int(row['htc_total_events']/1000)}k", 
                        (row['vehicles'], row['htc_total_events']), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            ax1.annotate(f"{int(row['ref_total_events']/1000)}k", 
                        (row['vehicles'], row['ref_total_events']), 
                        textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
        
        # 2. Diferença absoluta de eventos
        ax2 = axes[0, 1]
        bars = ax2.bar(df['vehicles'], df['total_diff'], alpha=0.7, color='coral')
        ax2.set_xlabel('Número de Veículos')
        ax2.set_ylabel('Diferença de Eventos (Interscs - HTC)')
        ax2.set_title('Diferença Absoluta de Eventos')
        ax2.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, diff in zip(bars, df['total_diff']):
            height = bar.get_height()
            ax2.annotate(f'{int(diff):,}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Eventos por veículo
        ax3 = axes[1, 0]
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, df['events_per_vehicle_htc'], width, label='HTC', alpha=0.8)
        bars2 = ax3.bar(x + width/2, df['events_per_vehicle_ref'], width, label='Interscsimulator', alpha=0.8)
        
        ax3.set_xlabel('Cenário')
        ax3.set_ylabel('Eventos por Veículo')
        ax3.set_title('Eventos por Veículo por Cenário')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{int(v)}' for v in df['vehicles']])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Fator de crescimento da diferença
        ax4 = axes[1, 1]
        base_diff = df[df['vehicles'] == 1000]['total_diff'].iloc[0]
        growth_factors = df['total_diff'] / base_diff
        expected_growth = df['vehicles'] / 1000  # Linear esperado
        
        ax4.plot(df['vehicles'], growth_factors, 'ro-', label='Crescimento Real', linewidth=2, markersize=8)
        ax4.plot(df['vehicles'], expected_growth, 'b--', label='Crescimento Linear Esperado', linewidth=2)
        ax4.set_xlabel('Número de Veículos')
        ax4.set_ylabel('Fator de Crescimento (relativo a 1000 veículos)')
        ax4.set_title('Crescimento da Diferença de Eventos')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Adicionar valores nos pontos
        for _, row in df.iterrows():
            vehicles = row['vehicles']
            factor = row['total_diff'] / base_diff
            ax4.annotate(f'{factor:.1f}x', 
                        (vehicles, factor), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Salvar gráfico
        output_path = self.output_dir / 'scalability_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {output_path}")
        
        plt.show()
    
    def generate_scalability_report(self, df):
        """Gera relatório detalhado de escalabilidade"""
        
        report_path = self.output_dir / 'scalability_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Análise de Escalabilidade dos Simuladores\n\n")
            f.write("## Resumo Executivo\n\n")
            
            # Calcular estatísticas
            base_1000 = df[df['vehicles'] == 1000].iloc[0]
            max_scenario = df[df['vehicles'] == 7500].iloc[0]
            
            base_diff = base_1000['total_diff']
            max_diff = max_scenario['total_diff']
            growth_factor = max_diff / base_diff if base_diff != 0 else 0
            
            f.write(f"**🚨 PROBLEMA IDENTIFICADO:** Crescimento não-linear da diferença de eventos\n\n")
            f.write(f"- **Baseline (1000 veículos):** {int(base_diff):,} eventos de diferença\n")
            f.write(f"- **Máximo (7500 veículos):** {int(max_diff):,} eventos de diferença\n")
            f.write(f"- **Fator de crescimento:** {growth_factor:.1f}x (esperado: 7.5x linear)\n")
            f.write(f"- **Excesso:** {(growth_factor - 7.5):.1f}x acima do esperado\n\n")
            
            f.write("## Dados Detalhados\n\n")
            f.write("| Veículos | HTC Eventos | Interscs Eventos | Diferença | Eventos/Veículo (HTC) | Eventos/Veículo (Interscs) |\n")
            f.write("|----------|-------------|------------------|-----------|----------------------|--------------------------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {int(row['vehicles']):,} | {int(row['htc_total_events']):,} | {int(row['ref_total_events']):,} | {int(row['total_diff']):,} | {row['events_per_vehicle_htc']:.1f} | {row['events_per_vehicle_ref']:.1f} |\n")
            
            f.write("\n## Análise dos Problemas\n\n")
            f.write("### 1. Crescimento Não-Proporcional\n\n")
            f.write("O número de eventos no Interscsimulator não cresce proporcionalmente ao número de veículos:\n\n")
            
            for _, row in df.iterrows():
                if row['vehicles'] == 1000:
                    continue
                vehicles = int(row['vehicles'])
                current_diff = row['total_diff']
                expected_diff = base_diff * (vehicles / 1000)
                excess_pct = ((current_diff - expected_diff) / expected_diff) * 100
                
                f.write(f"- **{vehicles:,} veículos:** {excess_pct:.1f}% acima do esperado\n")
            
            f.write("\n### 2. Possíveis Causas\n\n")
            f.write("#### Hipóteses para o Interscsimulator:\n")
            f.write("- **Congestão exponencial:** Conforme aumenta o número de veículos, o congestionamento pode causar:\n")
            f.write("  - Mais eventos de parada/retomada\n")
            f.write("  - Rebalanceamento de rotas mais frequente\n")
            f.write("  - Eventos adicionais de gerenciamento de densidade\n")
            f.write("- **Granularidade de simulação:** Maior detalhamento pode gerar eventos extras\n")
            f.write("- **Algoritmo de densidade:** Pode estar gerando eventos redundantes em alta densidade\n\n")
            
            f.write("### 3. Recomendações\n\n")
            f.write("1. **Investigar algoritmo de densidade** no Interscsimulator\n")
            f.write("2. **Analisar eventos por tipo** em cenários de alta densidade\n")
            f.write("3. **Verificar redundância de eventos** em situações de congestionamento\n")
            f.write("4. **Comparar tempo de simulação** entre os simuladores\n")
            f.write("5. **Validar se eventos extras representam comportamento real** ou overhead\n\n")
            
        print(f"📄 Relatório salvo em: {report_path}")
    
    def run_analysis(self):
        """Executa análise completa"""
        
        print("🚀 Iniciando análise de escalabilidade...")
        
        # Carregar dados
        self.load_comparison_data()
        
        if not self.data:
            print("❌ Nenhum dado encontrado!")
            return
        
        # Extrair métricas
        df = self.extract_event_metrics()
        
        # Análise detalhada
        self.analyze_scalability_issues(df)
        
        # Gerar gráficos
        self.create_scalability_plots(df)
        
        # Gerar relatório
        self.generate_scalability_report(df)
        
        print("\n✅ Análise de escalabilidade concluída!")

if __name__ == "__main__":
    analyzer = ScalabilityAnalyzer()
    analyzer.run_analysis()