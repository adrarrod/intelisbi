import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import matplotlib.pyplot as plt

@st.cache_resource
def ler_arquivo(arquivo):
    dados = pd.read_csv(arquivo, encoding='utf-8', sep=";", decimal=",")
    return dados

st.set_page_config(page_title="Intelisbi", layout="wide")
st.title("Sistema Intelisbi de análise exploratória")

with st.sidebar:
    arquivo = st.file_uploader("Escolha o arquivo:",type=['csv'],
                                   accept_multiple_files=False)
    process_button = st.button("Processar")
   
if process_button and arquivo is not None:
    try:
        dados = ler_arquivo(arquivo)
        dados['DataVenda'] = pd.to_datetime(dados['DataVenda'], format='%d/%m/%Y')
        dados['MesAno'] = dados['DataVenda'].dt.strftime('%m%Y')
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")    

if process_button and arquivo is not None:

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Indicadores","Dados Carregados","Localidade","Produtos","Categoria","Vendas Mensal","Cliente"])

    with tab1:
        quantidade_clientes = len(dados['CodigoCliente'].unique())
        conteudo1 = (f"Quantidade de Clientes em Carteira: {quantidade_clientes}")
        st.success(conteudo1)
        quantidade_clientes = len(dados['CodigoCliente'].unique())
        base_pedidos = pd.DataFrame(dados.groupby(['Pedido'])['ValorPedido'].count().sort_values(ascending=False))
        base_pedidos.rename(columns={'ValorPedido':'Valor Total Pedido'}, inplace=True)
        base_pedidos = base_pedidos.reset_index()
        ticket_medio_cliente = base_pedidos['Valor Total Pedido'].sum() / quantidade_clientes
        ticket_medio_cliente_formatado = f"{ticket_medio_cliente:.2f}".replace('.', ',')
        conteudo2 = (f"Valor de Ticket Médio: R${ticket_medio_cliente_formatado}")
        st.warning(conteudo2)

    with tab2:
    # Configurar GridOptions
        gb = GridOptionsBuilder.from_dataframe(dados)
        gb.configure_default_column(resizable=True, filterable=True)
        gb.configure_grid_options(fitColumnsOnGridLoad=True)
        gridOptions = gb.build()

        # Exibir os dados usando AgGrid com controle de altura e largura
        AgGrid(
            dados,
            gridOptions=gridOptions,
            theme='streamlit',
            width='1200px',  
            height=600    # Altura do grid em pixels
            )

    with tab3:
        clientes_por_cidade = dados.groupby('Cidade')['CodigoCliente'].nunique().reset_index()
        clientes_por_cidade = clientes_por_cidade.rename(columns={'CodigoCliente': 'QuantidadeClientes'})
        clientes_por_cidade = clientes_por_cidade.sort_values(by='QuantidadeClientes', ascending=False)
        plt.figure(figsize=(6, 2))
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.bar(clientes_por_cidade['Cidade'], clientes_por_cidade['QuantidadeClientes'], color='blue')
        ax.set_xlabel('Cidade', color="white")
        ax.set_ylabel('Quantidade de Clientes', color="white")
        ax.set_title('Quantidade de Clientes por Cidade', color="white")
        plt.xticks(rotation=45, ha='right', color="white")
        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)
        fig.patch.set_alpha(0.0)  # Fundo da figura
        ax.patch.set_alpha(0.0)   # Fundo dos eixos
        st.pyplot(fig)
    with tab4:
        produto = pd.DataFrame(dados.groupby(['NomeProduto'])['ValorPedido'].sum().sort_values(ascending=False))
        produto.rename(columns={'ValorPedido':'Valor Venda'}, inplace=True)
        produto = produto.reset_index()
        plt.figure(figsize=(6, 2))
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.bar(produto['NomeProduto'], produto['Valor Venda'], color='blue')
        #ax.set_xlabel('NomeProduto', color="white")
        #ax.set_ylabel('Produto', color="white")
        ax.set_title('Venda por Produto', color="white")
        plt.xticks(rotation=45, ha='right', color="white")
        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)
        fig.patch.set_alpha(0.0)  # Fundo da figura
        ax.patch.set_alpha(0.0)   # Fundo dos eixos
        st.pyplot(fig)
    with tab5:
        categoria = pd.DataFrame(dados.groupby(['Categoria'])['ValorPedido'].sum().sort_values(ascending=False))
        categoria.rename(columns={'ValorPedido':'Valor Venda'}, inplace=True)
        categoria = categoria.reset_index()
        categoria_cliente = pd.DataFrame(dados.groupby(['Categoria'])['CodigoCliente'].nunique().reset_index())
        categoria_cliente.rename(columns={'CodigoCliente':'Quantidade Cliente'}, inplace=True)
        df_combinado = pd.merge(categoria, categoria_cliente, on='Categoria')
        df_combinado['Ticket Médio'] = df_combinado['Valor Venda'] / df_combinado['Quantidade Cliente']
        plt.figure(figsize=(6, 2))
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.bar(df_combinado['Categoria'], df_combinado['Ticket Médio'], color='blue')
        #ax.set_xlabel('NomeProduto', color="white")
        #ax.set_ylabel('Produto', color="white")
        ax.set_title('Ticket Médio por Categoria', color="white")
        plt.xticks(rotation=45, ha='right', color="white")
        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)
        fig.patch.set_alpha(0.0)  # Fundo da figura
        ax.patch.set_alpha(0.0)   # Fundo dos eixos
        st.pyplot(fig)
    with tab6:
        venda_mensal = pd.DataFrame(dados.groupby(['MesAno'])['ValorPedido'].sum().reset_index())
        venda_mensal.rename(columns={'ValorPedido':'Valor Venda Mensal'}, inplace=True)
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.bar(venda_mensal['MesAno'], venda_mensal['Valor Venda Mensal'], color='blue')
        #ax.set_xlabel('NomeProduto', color="white")
        #ax.set_ylabel('Produto', color="white")
        ax.set_title('Valor de Venda Mensal', color="white")
        plt.xticks(rotation=45, ha='right', color="white")
        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)
        fig.patch.set_alpha(0.0)  # Fundo da figura
        ax.patch.set_alpha(0.0)   # Fundo dos eixos
        st.pyplot(fig)
    with tab7:
        gasto_por_cliente = pd.DataFrame(dados.groupby(['NomeCliente'])['ValorPedido'].sum().sort_values(ascending=False))
        gasto_por_cliente.rename(columns={'ValorPedido':'Valor Gasto por Cliente'}, inplace=True)
        gasto_por_cliente = gasto_por_cliente.reset_index()
        gasto_por_cliente_style = gasto_por_cliente.style.format({'Valor Gasto por Cliente': "R${:,.2f}"})
        st.dataframe(gasto_por_cliente_style)
