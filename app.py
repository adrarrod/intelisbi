import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from funcoes import *

# Configurar o estilo e a paleta de cores do Seaborn globalmente
sns.set_palette("colorblind")
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

if 'counter' not in st.session_state:
    st.session_state.counter = 0

st.set_page_config(page_title="Intelisbi", layout="wide")
st.title("Sistema Intelis BI de Análise Exploratória")

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Macro Indicadores", "Análise Localidade", "Análise Categoria", "Análise Produtos", "Análise Dia", "Vendas Mensal", "Análise Cliente","Recomendação","Segmentação"])

    with tab1:
        quantidade_clientes = len(dados['CodigoCliente'].unique())
        quantidade_vendas = len(dados['Pedido'].unique())
        base_pedidos = base_de_pedidos(dados)
        ticket_medio_cliente = base_pedidos['Valor Total Pedido'].sum() / quantidade_clientes
        ticket_medio_cliente = formatar_valor(ticket_medio_cliente)
        total_faturamento = base_dados_total_faturamento(dados)
        total_faturamento = formatar_valor(total_faturamento)
        quantidade_produtos_vendidos = base_dados_quantidade_produtos_vendidos(dados)
        ticket_medio_produto = base_dados_ticket_medio_produto(dados)
        ticket_medio_produto['Quantidade Venda Produto'] = ticket_medio_produto['Quantidade Venda Produto'].astype(str)
        gasto_por_cliente = base_dados_gasto_cliente(dados)
        quantidade_periodo = base_dados_periodo(dados) 
        produto = base_venda_produto_valor(dados)
        produto.set_index('NomeProduto', inplace=True)
        produto['Valor Venda'] = produto['Valor Venda'].apply(formatar_valor)
        produto_quantidade = base_dados_produto_quantidade(dados)
        produto_quantidade['Quantidade Venda Produto'] = produto_quantidade['Quantidade Venda Produto'].astype(str)
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Quantidade Clientes", quantidade_clientes, "#f0f0f0")
            metric_card("Total de Faturamento", total_faturamento, "#f0f0f0")
        with col2:
            metric_card("Valor Ticket Médio",ticket_medio_cliente, "#f0f0f0")
            metric_card("Quantidade Períodos/Mês", quantidade_periodo, "#f0f0f0")
        with col3:
            metric_card("Quantidade Vendas",quantidade_vendas, "#f0f0f0")
            metric_card("Produtos Vendidos",quantidade_produtos_vendidos, "#f0f0f0")
        st.write(" ")
        st.header("Dados de Produto")
        st.dataframe(ticket_medio_produto, use_container_width=True)

        st.write(" ")
        st.header("Top 5 Clientes")
        st.dataframe(gasto_por_cliente.head(5), use_container_width=True)

        st.write(" ")
        st.header("Top 5 Faturamento")
        st.dataframe(produto.head(5), use_container_width=True)

        st.write(" ")
        st.header("Top 5 Produto")
        st.dataframe(produto_quantidade.head(5), use_container_width=True)
        
        st.write(" ")
        st.header("Dados Gerais Carregados")
        st.dataframe(dados, use_container_width=True)

    with tab2:
        ticket_medio_cidade = base_dados_ticket_medio_produto_cidade(dados)
        ticket_medio_cidade['Quantidade Venda Produto'] = ticket_medio_cidade['Quantidade Venda Produto'].astype(str)
        st.dataframe(ticket_medio_cidade, use_container_width=True)

    with tab3:
        mes_ano_unicos = dados['MesAno'].unique()
        selecao_mes_ano = st.selectbox('Selecione o Mês/Ano', mes_ano_unicos, index=0, key='mes_ano_unicos')
        st.session_state.counter = 3
        col1, col2 = st.columns(2)
        with col1:
            vendas_por_categoria = bases_dados_categoria_quantidade(dados,selecao_mes_ano)
            fig, ax = grafico_por_categoria(vendas_por_categoria)
            st.pyplot(fig)
        with col2:
            df_combinado = base_categoria_ticket_medio(dados,selecao_mes_ano)
            fig, ax =  grafico_por_categoria_ticketmedio(df_combinado)
            st.pyplot(fig)

    with tab4:
        mes_ano_unicos = dados['MesAno'].unique()
        selecao_mes_ano = st.selectbox('Selecione o Mês/Ano', mes_ano_unicos, index=0, key='mes_ano_unicos1')
        st.session_state.counter = 4
        produto = base_venda_produto_valor_mes(dados,selecao_mes_ano)
        fig, ax  = grafico_venda_produto_valor(produto)
        st.pyplot(fig)

    with tab5:
        mes_ano_unicos = dados['MesAno'].unique()
        selecao_mes_ano = st.selectbox('Selecione o Mês/Ano', mes_ano_unicos, index=0, key='mes_ano_select')
        st.session_state.counter = 1
        venda_dia = base_dados_dia(dados,selecao_mes_ano)
        fig, ax = grafico_venda_dia(venda_dia)
        st.pyplot(fig)
        st.write(" ")
        vendas_diarias  = base_dados_produto_percentual(dados,selecao_mes_ano)
        fig = grafico_produto_percentual(vendas_diarias)
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
            nx.draw_networkx_nodes(G, pos, node_size=5000, node_color='green')
            nx.draw_networkx_edges(G, pos, width=[d['weight'] for (_, _, d) in G.edges(data=True)], alpha=0.6, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_size=15, font_color='white', font_weight='bold')  # Alterado para font_color='white'
            
            # Adicionar rótulos às arestas
            edge_labels = nx.get_edge_attributes(G, 'weight')
            edge_labels = {k: f"{v:.4f}" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color='black')  # Adicionado font_color='white'
            
            # Adicionar título e remover eixos
            plt.title('Regras de Associação de Produtos', fontsize=20, color='white')
            plt.axis('off')
            st.pyplot(plt)

        # Agrupar os dados por transações
        transacoes = dados.groupby(['Pedido', 'NomeProduto'])['Quantidade'].sum().unstack().reset_index().fillna(0).set_index('Pedido')
        transacoes = transacoes.applymap(lambda x: 1 if x > 0 else 0)

        # Ajustar parâmetros do Apriori
        min_support = 0.01
        min_threshold = 1.0

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

                pos = nx.spring_layout(G)

                # Desenhar o gráfico de rede
                draw_network(G, pos)

                # Adicionar representação textual das regras de associação
                st.subheader("Regras de Associação")
                total_lift_percentage = 0
                for _, row in top_regras.iterrows():
                    antecedents = ", ".join(list(row['antecedents']))
                    consequents = ", ".join(list(row['consequents']))
                    lift_percentage = (row['lift'] - 1) * 100
                    total_lift_percentage += lift_percentage
                    adjusted_lift_percentage = lift_percentage * 0.90
                    st.write(f"Se a pessoa comprar {antecedents}, então ela tem {adjusted_lift_percentage:.2f}% mais chance de comprar {consequents}.")
                
                # Exibir tabela de regras de associação com suporte
                #st.subheader("Tabela de Regras de Associação")
                #st.dataframe(top_regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

                #st.write(f"Suporte mínimo usado: {min_support}")

                #st.dataframe(top_regras)
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



