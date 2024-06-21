import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configurar o estilo e a paleta de cores do Seaborn globalmente
sns.set_palette("colorblind")

if 'counter' not in st.session_state:
    st.session_state.counter = 0

def formatar_valor(valor):
    valor_formatado = f"R$ {valor:,.2f}"
    valor_formatado = valor_formatado.replace(",", "v").replace(".", ",").replace("v", ".")
    return valor_formatado

def formatar_valor_sem_cifrao(valor):
    valor_formatado = f"{valor:,.2f}"
    valor_formatado = valor_formatado.replace(",", "v").replace(".", ",").replace("v", ".")
    return valor_formatado

def metric_card(title, value, background_color):
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; border: 1px solid #e6e6e6; border-radius: 10px; padding: 5px; margin: 5px 0; width: 100%; background-color: {background_color};">
        <div style="flex: 1; text-align: left;">
            <h4 style="margin: 0; font-size: 12px; color: #070707;">{title}</h4>
        </div>
        <div style="flex: 1; text-align: right;">
            <h4 style="margin: 0; font-size: 14px; color: #333;">{value}</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)

def calcular_variacao_percentual(dados):
    dados['Variação (%)'] = dados['ValorPedido'].pct_change() * 100
    return dados

def bases_dados_categoria_quantidade(dados):
    vendas_por_categoria = dados.groupby('Categoria')['Quantidade'].sum().reset_index()
    return vendas_por_categoria

def base_de_pedidos(dados):
        base_pedidos = pd.DataFrame(dados.groupby(['Pedido'])['ValorPedido'].count().sort_values(ascending=False))
        base_pedidos.rename(columns={'ValorPedido': 'Valor Total Pedido'}, inplace=True)
        base_pedidos = base_pedidos.reset_index()
        return base_pedidos

def base_categoria_ticket_medio(dados):
        categoria = pd.DataFrame(dados.groupby(['Categoria'])['ValorPedido'].sum().sort_values(ascending=False))
        categoria.rename(columns={'ValorPedido': 'Valor Venda'}, inplace=True)
        categoria = categoria.reset_index()
        categoria_cliente = pd.DataFrame(dados.groupby(['Categoria'])['CodigoCliente'].nunique().reset_index())
        categoria_cliente.rename(columns={'CodigoCliente': 'Quantidade Cliente'}, inplace=True)
        df_combinado = pd.merge(categoria, categoria_cliente, on='Categoria')
        df_combinado['Ticket Médio'] = df_combinado['Valor Venda'] / df_combinado['Quantidade Cliente']
        return df_combinado

def base_venda_produto_valor(dados):
        produto = pd.DataFrame(dados.groupby(['NomeProduto'])['ValorPedido'].sum().sort_values(ascending=False))
        produto.rename(columns={'ValorPedido': 'Valor Venda'}, inplace=True)
        produto = produto.reset_index()
        return produto

def base_venda_mensal(dados):
    venda_mensal = pd.DataFrame(dados.groupby(['MesAno'])['ValorPedido'].sum().reset_index())
    venda_mensal.rename(columns={'ValorPedido':'Valor Venda Mensal'}, inplace=True)
    return venda_mensal

def base_dados_dia(dados, anosmes):
    filtro_dia = dados.query(f"`MesAno` == '{anosmes}'")
    venda_dia = pd.DataFrame(filtro_dia.groupby(['DataVenda'])['CodigoCliente'].nunique().reset_index())
    venda_dia.rename(columns={'CodigoCliente':'Quantidade Cliente'}, inplace=True)
    return venda_dia

def base_dados_gasto_cliente(dados):
    gasto_por_cliente = pd.DataFrame(dados.groupby(['NomeCliente'])['ValorPedido'].sum().sort_values(ascending=False))
    gasto_por_cliente.rename(columns={'ValorPedido': 'Valor Gasto por Cliente'}, inplace=True)
    gasto_por_cliente = gasto_por_cliente.reset_index()
    gasto_por_cliente['Valor Gasto por Cliente'] = gasto_por_cliente['Valor Gasto por Cliente'].apply(formatar_valor)
    return gasto_por_cliente

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
        ax.set_xlabel('Categoria', color='white')
        ax.set_ylabel('Quantidade Vendida', color='white')
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
        ax.set_xlabel('Categoria', color='white')
        ax.set_ylabel('Ticket Médio', color='white')
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
        ax.set_xlabel('Produto', color='white')
        ax.set_ylabel('Valor Venda', color='white')
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

    # Gráfico de linha para valor vendido
    sns.lineplot(x='MesAno', y='Valor Venda Mensal', data=vendas_por_mes, marker='o', ax=ax1)
    ax1.set_xlabel('Mês/Ano')
    ax1.set_ylabel('Valor Vendido (R$)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Formatando os valores no eixo y
    labels = [formatar_valor(valor) for valor in ax1.get_yticks()]
    ax1.set_yticklabels(labels)

    # Ajustar os rótulos do eixo x
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Aumenta o espaço na parte inferior para evitar sobreposição

    plt.title('Vendas por Mês')
    plt.tight_layout()

    return fig, ax1

def grafico_venda_dia(venda_dia):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y='DataVenda', x='Quantidade Cliente', data=venda_dia, ax=ax, palette="colorblind", edgecolor='none', linewidth=0.3, width=0.3)
        
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
        
        ax.set_title('Venda por Dia', color='white')
        ax.set_ylabel('Data Venda', color='white')
        ax.set_xlabel('Quantidade Cliente', color='white')
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

@st.cache_resource
def ler_arquivo(arquivo):
    dados = pd.read_csv(arquivo, encoding='utf-8', sep=";", decimal=",")
    return dados

st.set_page_config(page_title="Intelisbi", layout="wide")
st.title("Sistema Intelisbi de Análise Exploratória")

with st.sidebar:
    arquivo = st.file_uploader("Escolha o arquivo:", type=['csv'], accept_multiple_files=False)
    process_button = st.button("Processar")


if st.session_state.counter >= 1:
    process_button = True

if process_button and arquivo is not None:
    try:
        dados = ler_arquivo(arquivo)
        dados['DataVenda'] = pd.to_datetime(dados['DataVenda'], format='%d/%m/%Y')
        dados['MesAno'] = dados['DataVenda'].dt.strftime('%m%Y')
        dados['Dia'] = dados['DataVenda'].dt.day
        dados['Mês'] = dados['DataVenda'].dt.month
        dados['Ano'] = dados['DataVenda'].dt.year
        dados['ValorProduto'] = dados['ValorProduto'].astype(str).str.replace(',', '.').astype(float)
        dados['ValorPedido'] = dados['ValorPedido'].astype(str).str.replace(',', '.').astype(float)
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")    

if process_button and arquivo is not None:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Indicadores", "Dados Carregados", "Análise Categoria", "Análise Produtos", "Pedidos Dia", "Vendas Mensal", "Análise Cliente","Recomendação","Segmentação"])

    with tab1:
        quantidade_clientes = len(dados['CodigoCliente'].unique())
        quantidade_vendas = len(dados['Pedido'].unique())
        base_pedidos = base_de_pedidos(dados)
        ticket_medio_cliente = base_pedidos['Valor Total Pedido'].sum() / quantidade_clientes
        ticket_medio_cliente_formatado = f"{ticket_medio_cliente:.2f}".replace('.', ',')
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Quantidade Clientes", quantidade_clientes, "#f0f0f0")
        with col2:
            metric_card("Valor de Ticket Médio",ticket_medio_cliente_formatado, "#f0f0f0")
        with col3:
            metric_card("Quantidade Vendas",quantidade_vendas, "#f0f0f0")

    with tab2:
        st.dataframe(dados)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            vendas_por_categoria = bases_dados_categoria_quantidade(dados)
            fig, ax = grafico_por_categoria(vendas_por_categoria)
            st.pyplot(fig)
        with col2:
            df_combinado = base_categoria_ticket_medio(dados)
            fig, ax =  grafico_por_categoria_ticketmedio(df_combinado)
            st.pyplot(fig)

    with tab4:
        produto = base_venda_produto_valor(dados)
        fig, ax  = grafico_venda_produto_valor(produto)
        st.pyplot(fig)

    with tab5:
        mes_ano_unicos = dados['MesAno'].unique()
        selecao_mes_ano = st.selectbox('Selecione o Mês/Ano', mes_ano_unicos, index=0, key='mes_ano_select')
        st.session_state.counter = 1
        venda_dia = base_dados_dia(dados,selecao_mes_ano)
        fig, ax = grafico_venda_dia(venda_dia)
        st.pyplot(fig)

    with tab6:
        venda_mensal = base_venda_mensal(dados)
        fig, ax1 = grafico_venda_mensal(venda_mensal)
        st.pyplot(fig)

    with tab7:
        gasto_por_cliente = base_dados_gasto_cliente(dados)
        st.dataframe(gasto_por_cliente)

    with tab8:
        def draw_network(G, pos):
            plt.figure(figsize=(14, 6))
            nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue')
            nx.draw_networkx_edges(G, pos, width=[d['weight'] for (_, _, d) in G.edges(data=True)], alpha=0.6, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_size=15, font_color='black', font_weight='bold')
            
            # Adicionar rótulos às arestas
            edge_labels = nx.get_edge_attributes(G, 'weight')
            edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
            
            # Adicionar título e remover eixos
            plt.title('Regras de Associação de Produtos', fontsize=20)
            plt.axis('off')
            st.pyplot(plt)

        # Agrupar os dados por transações
        transacoes = dados.groupby(['Pedido', 'NomeProduto'])['Quantidade'].sum().unstack().reset_index().fillna(0).set_index('Pedido')
        transacoes = transacoes.applymap(lambda x: 1 if x > 0 else 0)

        # Ajustar parâmetros do Apriori
        min_support = st.sidebar.slider('min_support', 0.01, 0.5, 0.01)
        min_threshold = st.sidebar.slider('min_threshold', 1.0, 10.0, 1.0)

        # Aplicar o algoritmo Apriori
        frequent_itemsets = apriori(transacoes, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            st.warning("Nenhum itemset frequente encontrado. Tente ajustar o parâmetro de suporte mínimo.")
        else:
            regras = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
            if regras.empty:
                st.warning("Nenhuma regra encontrada. Tente ajustar os parâmetros de suporte e limiar de lift.")
            else:
                # Selecionar as regras mais fortes para visualização simplificada
                top_regras = regras.nlargest(15, 'lift')

                # Criar o gráfico de rede
                G = nx.DiGraph()

                for _, row in top_regras.iterrows():
                    for antecedent in row['antecedents']:
                        for consequent in row['consequents']:
                            G.add_edge(antecedent, consequent, weight=row['lift'])

                # Posição dos nós
                pos = nx.spring_layout(G, seed=42)

                # Desenhar o gráfico de rede
                draw_network(G, pos)

                # Adicionar representação textual das regras de associação
                st.subheader("Regras de Associação")
                for _, row in top_regras.iterrows():
                    antecedents = ", ".join(list(row['antecedents']))
                    consequents = ", ".join(list(row['consequents']))
                    lift_percentage = (row['lift'] - 1) * 100
                    st.write(f"Se a pessoa comprar {antecedents}, então ela tem {lift_percentage:.2f}% mais chance de comprar {consequents}.")
                
                # Exibir tabela de regras de associação com suporte
                st.subheader("Tabela de Regras de Associação")
                st.dataframe(top_regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

                st.write(f"Suporte mínimo usado: {min_support}")

                st.dataframe(regras)
    with tab9:
        # Agregar dados de compras por cliente
        cliente_compras = dados.groupby('CodigoCliente').agg({
            'ValorPedido': 'sum',
            'Pedido': 'count',
            'Quantidade': 'sum'
        }).reset_index()

        # Renomear colunas para clareza
        cliente_compras.columns = ['CodigoCliente', 'ValorTotal', 'NumPedidos', 'QuantidadeTotal']

        # Padronizar os dados
        scaler = StandardScaler()
        cliente_compras_scaled = scaler.fit_transform(cliente_compras[['ValorTotal', 'NumPedidos', 'QuantidadeTotal']])

        num_clusters = 4
        # Aplicar K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cliente_compras['Cluster'] = kmeans.fit_predict(cliente_compras_scaled)   

        # Gráficos de visualização
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        sns.scatterplot(data=cliente_compras, x='ValorTotal', y='NumPedidos', hue='Cluster', palette='viridis', ax=ax[0])
        ax[0].set_title('Valor Total vs Número de Pedidos')
        ax[0].set_xlabel('Valor Total')
        ax[0].set_ylabel('Número de Pedidos')

        sns.scatterplot(data=cliente_compras, x='ValorTotal', y='QuantidadeTotal', hue='Cluster', palette='viridis', ax=ax[1])
        ax[1].set_title('Valor Total vs Quantidade Total')
        ax[1].set_xlabel('Valor Total')
        ax[1].set_ylabel('Quantidade Total')

        sns.scatterplot(data=cliente_compras, x='NumPedidos', y='QuantidadeTotal', hue='Cluster', palette='viridis', ax=ax[2])
        ax[2].set_title('Número de Pedidos vs Quantidade Total')
        ax[2].set_xlabel('Número de Pedidos')
        ax[2].set_ylabel('Quantidade Total')

        plt.legend(title='Cluster')
        plt.tight_layout() 

        st.pyplot(fig)   

        clientes = dados[['CodigoCliente', 'NomeCliente']].drop_duplicates()
        cliente_compras_com_nome = pd.merge(cliente_compras, clientes, on='CodigoCliente', how='left')
        segmentos = [0,1,2,3]
        cluster = st.selectbox('Selecione a segmentação: ', segmentos, index=0, key='cluster')
        st.session_state.counter = 2
        segemento_filtrado = cliente_compras_com_nome.query(f"`Cluster` == {cluster}")
        segemento_filtrado['ValorTotal'] = segemento_filtrado['ValorTotal'].apply(formatar_valor)
        st.dataframe(segemento_filtrado)



