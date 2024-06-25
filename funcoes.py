import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def formatar_valor(valor):
    valor_formatado = f"R$ {valor:,.2f}"
    valor_formatado = valor_formatado.replace(",", "v").replace(".", ",").replace("v", ".")
    return valor_formatado

def formatar_milhar(valor):
    if isinstance(valor, (int, float)):
        return f'{valor:,.0f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    return valor

def formatar_valor_sem_cifrao(valor):
    valor_formatado = f"{valor:,.2f}"
    valor_formatado = valor_formatado.replace(",", "v").replace(".", ",").replace("v", ".")
    return valor_formatado

def metric_card(title, value, background_color):
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; border: 1px solid #e6e6e6; border-radius: 10px; padding: 5px; margin: 5px 0; width: 100%; background-color: {background_color};">
        <div style="flex: 1; text-align: left;">
            <h4 style="margin: 0; font-size: 18px; color: #070707;">{title}</h4>
        </div>
        <div style="flex: 1; text-align: right;">
            <h4 style="margin: 0; font-size: 18px; color: #333;">{value}</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)

def calcular_variacao_percentual(dados):
    dados['Variação (%)'] = dados['ValorPedido'].pct_change() * 100
    return dados

def bases_dados_categoria_quantidade(dados,anosmes):
    filtro_mes = dados.query(f"`MesAno` == '{anosmes}'")
    vendas_por_categoria = filtro_mes.groupby('Categoria')['Quantidade'].sum().reset_index()
    vendas_por_categoria = vendas_por_categoria.sort_values(by='Quantidade', ascending=False)
    return vendas_por_categoria

def base_de_pedidos(dados):
        base_pedidos = pd.DataFrame(dados.groupby(['Pedido'])['ValorPedido'].count().sort_values(ascending=False))
        base_pedidos.rename(columns={'ValorPedido': 'Valor Total Pedido'}, inplace=True)
        base_pedidos = base_pedidos.reset_index()
        return base_pedidos

def base_categoria_ticket_medio(dados,anosmes):
        filtro_mes = dados.query(f"`MesAno` == '{anosmes}'")
        categoria = pd.DataFrame(filtro_mes.groupby(['Categoria'])['ValorPedido'].sum().sort_values(ascending=False))
        categoria.rename(columns={'ValorPedido': 'Valor Venda'}, inplace=True)
        categoria = categoria.reset_index()
        categoria_cliente = pd.DataFrame(filtro_mes.groupby(['Categoria'])['CodigoCliente'].nunique().reset_index())
        categoria_cliente.rename(columns={'CodigoCliente': 'Quantidade Cliente'}, inplace=True)
        df_combinado = pd.merge(categoria, categoria_cliente, on='Categoria')
        df_combinado['Ticket Médio'] = df_combinado['Valor Venda'] / df_combinado['Quantidade Cliente']
        df_combinado = df_combinado.sort_values(by='Ticket Médio', ascending=False)
        return df_combinado

def base_venda_produto_valor(dados):
        produto = pd.DataFrame(dados.groupby(['NomeProduto'])['ValorPedido'].sum().sort_values(ascending=False))
        produto.rename(columns={'ValorPedido': 'Valor Venda'}, inplace=True)
        produto = produto.reset_index()
        return produto

def base_venda_produto_valor_mes(dados,anosmes):
        filtro_mes = dados.query(f"`MesAno` == '{anosmes}'")
        produto = pd.DataFrame(filtro_mes.groupby(['NomeProduto'])['ValorPedido'].sum().sort_values(ascending=False))
        produto.rename(columns={'ValorPedido': 'Valor Venda'}, inplace=True)
        produto = produto.reset_index()
        return produto

def base_venda_mensal(dados):
    venda_mensal = pd.DataFrame(dados.groupby(['MesAno'])['ValorPedido'].sum().reset_index())
    venda_mensal.rename(columns={'ValorPedido':'Valor Venda Mensal'}, inplace=True)
    return venda_mensal

def base_dados_dia(dados, anosmes):
    filtro_dia = dados.query(f"`MesAno` == '{anosmes}'")
    venda_dia = pd.DataFrame(filtro_dia.groupby(['DataVenda'])['Quantidade'].sum().reset_index())
    venda_dia.rename(columns={'Quantidade':'Quantidade Produto'}, inplace=True)
    return venda_dia

def base_dados_gasto_cliente(dados):
    gasto_por_cliente = pd.DataFrame(dados.groupby(['NomeCliente'])['ValorPedido'].sum().sort_values(ascending=False))
    gasto_por_cliente.rename(columns={'ValorPedido': 'Valor Gasto por Cliente'}, inplace=True)
    gasto_por_cliente = gasto_por_cliente.reset_index()
    gasto_por_cliente['Valor Gasto por Cliente'] = gasto_por_cliente['Valor Gasto por Cliente'].apply(formatar_valor)
    gasto_por_cliente.set_index('NomeCliente', inplace=True)
    return gasto_por_cliente

def base_dados_total_faturamento(dados):
    total_faturamento = dados['ValorPedido'].sum()
    return total_faturamento

def base_dados_quantidade_produtos(dados):
    quantidade_produtos = dados['CodigoProduto'].nunique()
    return quantidade_produtos

def base_dados_quantidade_produtos_vendidos(dados):
    quantidade_produtos_vendidos = dados['Quantidade'].sum()
    return quantidade_produtos_vendidos

def base_dados_ticket_medio_produto(dados):
        produto_valor = pd.DataFrame(dados.groupby(['NomeProduto'])['ValorPedido'].sum().sort_values(ascending=False))
        produto_valor.rename(columns={'ValorPedido': 'Valor Venda Produto'}, inplace=True)
        produto_valor = produto_valor.reset_index()
        produto_quantidade = pd.DataFrame(dados.groupby(['NomeProduto'])['Quantidade'].count().sort_values(ascending=False))
        produto_quantidade.rename(columns={'Quantidade': 'Quantidade Venda Produto'}, inplace=True)
        produto_quantidade = produto_quantidade.reset_index()
        df_combinado = pd.merge(produto_valor, produto_quantidade, on='NomeProduto')
        df_combinado['Ticket Médio Produto'] = df_combinado['Valor Venda Produto'] / df_combinado['Quantidade Venda Produto']
        df_combinado['Ticket Médio Produto'] = df_combinado['Ticket Médio Produto'].apply(formatar_valor)
        df_combinado['Valor Venda Produto'] = df_combinado['Valor Venda Produto'].apply(formatar_valor)
        df_combinado.set_index('NomeProduto', inplace=True)
        return df_combinado

def base_dados_periodo(dados):
     quantidade_periodo = dados['MesAno'].nunique()
     return quantidade_periodo

def base_dados_produto_quantidade(dados):
     produto_quantidade = pd.DataFrame(dados.groupby(['NomeProduto'])['Quantidade'].count().sort_values(ascending=False))
     produto_quantidade.rename(columns={'Quantidade': 'Quantidade Venda Produto'}, inplace=True)
     produto_quantidade = produto_quantidade.reset_index()
     produto_quantidade.set_index('NomeProduto', inplace=True)
     produto_quantidade['Quantidade Venda Produto'] = produto_quantidade['Quantidade Venda Produto'].apply(formatar_milhar)
     return produto_quantidade

def base_dados_ticket_medio_produto_cidade(dados):
    produto_valor = pd.DataFrame(dados.groupby(['NomeProduto','Cidade'])['ValorPedido'].sum().sort_values(ascending=False))
    produto_valor.rename(columns={'ValorPedido': 'Valor Venda Produto'}, inplace=True)
    produto_valor = produto_valor.reset_index()
    produto_quantidade = pd.DataFrame(dados.groupby(['NomeProduto','Cidade'])['Quantidade'].count().sort_values(ascending=False))
    produto_quantidade.rename(columns={'Quantidade': 'Quantidade Venda Produto'}, inplace=True)
    produto_quantidade = produto_quantidade.reset_index()
    df_combinado = pd.merge(produto_valor, produto_quantidade, on=['NomeProduto','Cidade'])
    df_combinado['Ticket Médio Produto'] = df_combinado['Valor Venda Produto'] / df_combinado['Quantidade Venda Produto']
    df_combinado['Ticket Médio Produto'] = df_combinado['Ticket Médio Produto'].apply(formatar_valor)
    df_combinado['Valor Venda Produto'] = df_combinado['Valor Venda Produto'].apply(formatar_valor)
    df_combinado.set_index('NomeProduto', inplace=True)
    return df_combinado

def base_dados_produto_percentual(dados,anosmes):
    # Converter a coluna DataVenda para o formato datetime
    dados['DataVenda'] = pd.to_datetime(dados['DataVenda'], format='%d/%m/%Y')
    filtro_mes = dados.query(f"`MesAno` == '{anosmes}'")
    # Agrupar por dia e nome do produto e calcular a quantidade de vendas
    vendas_diarias = filtro_mes.groupby([filtro_mes['DataVenda'].dt.date, 'NomeProduto'])['Quantidade'].sum().reset_index()
    vendas_diarias = vendas_diarias.rename(columns={'DataVenda': 'Dia'})
    return vendas_diarias

def grafico_por_categoria(vendas_por_categoria):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Categoria', y='Quantidade', data=vendas_por_categoria, ax=ax, palette="colorblind", width=0.4, edgecolor='none')
        
        for p in ax.patches:
            # Criar um retângulo com bordas arredondadas
            rounded_rect = Rectangle((p.get_x(), 0), p.get_width(), p.get_height(), 
                                     linewidth=0, edgecolor='none', facecolor=p.get_facecolor(), 
                                     joinstyle="round", antialiased=True, zorder=10)
            ax.add_patch(rounded_rect)
            p.remove()  # Remover a barra original
            ax.annotate(f'{int(rounded_rect.get_height())}', 
                        (rounded_rect.get_x() + rounded_rect.get_width() / 2., rounded_rect.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points', 
                        color='white')  # Texto branco
        
        ax.set_title('Vendas por Categoria de Produto', color='white')
        #ax.set_xlabel('Categoria', color='white')
        #ax.set_ylabel('Quantidade Vendida', color='white')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(colors='white')  # Altera a cor dos ticks e seus labels
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')
        fig.patch.set_facecolor('black')  # Define a cor de fundo da figura
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.patch.set_facecolor('black')   # Define a cor de fundo dos eixos
        ax.grid(False)
        plt.tight_layout()
        
    return fig, ax

def grafico_por_categoria_ticketmedio(df_combinado):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Categoria', y='Ticket Médio', data=df_combinado, ax=ax, palette="colorblind", width=0.4, edgecolor='none')
        
        for p in ax.patches:
            # Criar um retângulo com bordas arredondadas
            rounded_rect = Rectangle((p.get_x(), 0), p.get_width(), p.get_height(), 
                                     linewidth=0, edgecolor='none', facecolor=p.get_facecolor(), 
                                     joinstyle="round", antialiased=True, zorder=10)
            ax.add_patch(rounded_rect)
            p.remove()  # Remover a barra original
            ax.annotate(formatar_valor(rounded_rect.get_height()), 
                        (rounded_rect.get_x() + rounded_rect.get_width() / 2., rounded_rect.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points', 
                        color='white')  # Texto branco
        
        ax.set_title('Ticket Médio por Categoria', color='white')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(colors='white')  # Altera a cor dos ticks e seus labels
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')
        fig.patch.set_facecolor('black')  # Define a cor de fundo da figura
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.patch.set_facecolor('black')   # Define a cor de fundo dos eixos
        ax.grid(False)
        plt.tight_layout()
        
    return fig, ax

def grafico_venda_produto_valor(produto):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='NomeProduto', y='Valor Venda', data=produto, ax=ax, palette="colorblind", width=0.4, edgecolor='none')
        
        for p in ax.patches:
            # Criar um retângulo com bordas arredondadas
            rounded_rect = Rectangle((p.get_x(), 0), p.get_width(), p.get_height(), 
                                     linewidth=0, edgecolor='none', facecolor=p.get_facecolor(), 
                                     joinstyle="round", antialiased=True, zorder=10)
            ax.add_patch(rounded_rect)
            p.remove()  # Remover a barra original
            ax.annotate(formatar_valor_sem_cifrao(rounded_rect.get_height()), 
                        (rounded_rect.get_x() + rounded_rect.get_width() / 2., rounded_rect.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points', 
                        color='white',
                        fontsize=8)  
        
        ax.set_title('Venda por Produto', color='white')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(colors='white')  # Altera a cor dos ticks e seus labels
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')
        fig.patch.set_facecolor('black')  # Define a cor de fundo da figura
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.patch.set_facecolor('black')   # Define a cor de fundo dos eixos
        ax.grid(False)
        plt.tight_layout()
        
    return fig, ax    

def grafico_venda_mensal(vendas_por_mes):
    vendas_por_mes['MesAno'] = pd.to_datetime(vendas_por_mes['MesAno'], format='%m%Y')
    vendas_por_mes = vendas_por_mes.sort_values('MesAno')
    vendas_por_mes['MesAno'] = vendas_por_mes['MesAno'].dt.strftime('%m/%Y')

    # Configurar o gráfico com linha e variação percentual apenas com os meses
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Definir o fundo preto para o gráfico
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')

    # Gráfico de linha para valor vendido
    sns.lineplot(x='MesAno', y='Valor Venda Mensal', data=vendas_por_mes, marker='o', ax=ax1, color='red')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.tick_params(axis='y', labelcolor='white')
    ax1.tick_params(axis='x', labelcolor='white')

    # Formatando os valores no eixo y
    labels = [formatar_valor(valor) for valor in ax1.get_yticks()]
    ax1.set_yticklabels(labels)

    # Ajustar os rótulos do eixo x
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Aumenta o espaço na parte inferior para evitar sobreposição

    # Adicionar grade com linhas brancas
    ax1.grid(True, which='both', color='white', linestyle='--', linewidth=0.5)

    plt.title('Vendas por Mês')
    plt.tight_layout()

    return fig, ax1

def grafico_venda_dia(venda_dia):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y='DataVenda', x='Quantidade Produto', data=venda_dia, ax=ax, palette="colorblind", edgecolor='none', linewidth=0.3, width=0.3)
        
        for p in ax.patches:
            # Criar um retângulo com bordas arredondadas
            rounded_rect = Rectangle((0, p.get_y()), p.get_width(), p.get_height(), 
                                     linewidth=0, edgecolor='none', facecolor=p.get_facecolor(), 
                                     joinstyle="round", antialiased=True, zorder=10)
            ax.add_patch(rounded_rect)
            p.remove()  # Remover a barra original
            ax.annotate(f'{int(rounded_rect.get_width())}', 
                        (rounded_rect.get_width(), rounded_rect.get_y() + rounded_rect.get_height() / 2.), 
                        ha='left', va='center', 
                        xytext=(5, 0), 
                        textcoords='offset points', 
                        color='white')  # Texto branco
        
        ax.set_title('Venda por Dia / Quantidade Produto', color='white')
        #ax.set_ylabel('Data Venda', color='white')
        #ax.set_xlabel('Quantidade Produto', color='white')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(colors='white')  # Altera a cor dos ticks e seus labels
        plt.xticks(color='white')
        plt.yticks(color='white')
        fig.patch.set_facecolor('black')  # Define a cor de fundo da figura
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.patch.set_facecolor('black')   # Define a cor de fundo dos eixos
        ax.grid(False)
        plt.tight_layout()
        
    return fig, ax

def grafico_produto_percentual(vendas_diarias):
    with sns.axes_style("darkgrid"):
        # Calcular o total de vendas por dia
        total_vendas_diarias = vendas_diarias.groupby('Dia')['Quantidade'].sum().reset_index()
        total_vendas_diarias = total_vendas_diarias.rename(columns={'Quantidade': 'TotalVendas'})
        
        # Juntar os dados para calcular o percentual
        vendas_percentual = vendas_diarias.merge(total_vendas_diarias, on='Dia')
        vendas_percentual['Percentual'] = (vendas_percentual['Quantidade'] / vendas_percentual['TotalVendas']) * 100
        
        # Pivotar os dados para o formato necessário para plotagem empilhada
        pivot_data = vendas_percentual.pivot(index='Dia', columns='NomeProduto', values='Percentual').fillna(0)
        
        # Plotar o gráfico de barras empilhadas
        fig, ax = plt.subplots(figsize=(14, 7))
        pivot_data.plot(kind='bar', stacked=True, colormap='tab20', ax=ax, width=0.9)
        # Configurar o fundo do gráfico e dos eixos
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('Percentual de Produtos Vendidos por Dia')
        ax.legend(title='Produto', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 100)
        plt.tight_layout()

        for container in ax.containers:
            # Criar rótulos, excluindo valores 0.0%
            labels = [f'{v:.1f}%' if v > 0 else '' for v in container.datavalues]
            ax.bar_label(container, labels=labels, label_type='center', color='black', fontsize=7)

        # Ajustar os limites do eixo y para adicionar espaço extra no topo
        ax.set_ylim(0, 105)

        return fig

@st.cache_resource
def ler_arquivo(arquivo):
    dados = pd.read_csv(arquivo, encoding='utf-8', sep=";", decimal=",")
    return dados
