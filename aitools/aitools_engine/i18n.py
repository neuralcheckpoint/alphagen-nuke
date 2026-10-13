"""User-facing text in the system language: Portuguese or English (the fallback).

Standard library only: the Nuke plugins, the engines and the Medratools installer all import it.
MEDRATOOLS_LANG (e.g. "pt" or "en") overrides the detected language.
"""

import locale
import os


def language():
    code = os.environ.get("MEDRATOOLS_LANG", "")
    if not code and os.name == "nt":
        import ctypes

        code = locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), "")
    for name in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        if code:
            break
        code = os.environ.get(name, "")
    return code.split(":")[0][:2].lower()


PT = {
    # Nuke panel
    "Input": "Entrada",
    "Subjects": "Sujeitos",
    "Advanced": "Avançado",
    "About": "Sobre",
    "Without points Alpha Gen picks the subject itself.<br>"
    "To choose: Analyze people once (~15 s), then drag a point onto each person to keep. "
    "Tick exclude to drop one.<br>"
    "A point off the people (a prop, someone missed) is a SAM click: check it with Preview selection (red).":
        "Sem pontos, o Alpha Gen escolhe o sujeito sozinho.<br>"
        "Para escolher: Analisar pessoas uma vez (~15 s) e arraste um ponto sobre cada pessoa a manter. "
        "Marque excluir para tirar uma.<br>"
        "Um ponto fora das pessoas (um objeto, alguém não encontrado) é um clique do SAM: confira com Prévia da seleção (vermelho).",
    "LTX-2.5 Alpha Gen matte; apply it as the alpha of the original plate.<br>"
    "Subject points need the optional SAM 2.1 setup (~1.2 GB, separate CUDA environment). "
    "The SAM mask only crops the matte; it cannot recover a missed subject.<br>"
    "Long shots run in overlapping chunks of up to 121 frames with blended boundaries.":
        "Matte do LTX-2.5 Alpha Gen; use-o como alpha do plate original.<br>"
        "Os pontos de sujeito precisam da instalação opcional do SAM 2.1 (~1,2 GB, ambiente CUDA separado). "
        "A máscara do SAM só recorta o matte; não recupera um sujeito perdido.<br>"
        "Planos longos rodam em blocos sobrepostos de até 121 quadros, com transições mescladas.",
    "output": "saída",
    "processing resolution": "resolução de processamento",
    "source": "origem",
    "frame range": "intervalo de quadros",
    "from input": "da entrada",
    "status": "estado",
    "not computed": "não calculado",
    "Compute": "Calcular",
    "Recompute all": "Recalcular tudo",
    "Subject {}": "Sujeito {}",
    "remove": "remover",
    "subjects": "sujeitos",
    "+ subject": "+ sujeito",
    "reset subjects": "limpar sujeitos",
    "Drag the point in the Viewer; its frame is saved when you move it.":
        "Arraste o ponto no Viewer; o quadro atual é salvo quando você o mover.",
    "Drag a point onto each subject in the Viewer. No active point keeps automatic selection.":
        "Arraste um ponto até cada sujeito no Viewer. Sem ponto ativo, a seleção continua automática.",
    "Add another subject point.": "Adiciona outro ponto de sujeito.",
    "Clear all points; Alpha Gen returns to automatic subject selection.":
        "Limpa todos os pontos; o Alpha Gen volta à seleção automática do sujeito.",
    "Subjects": "Sujeitos",
    "point {}": "ponto {}",
    "subject": "sujeito",
    "exclude": "excluir",
    "+ point": "+ ponto",
    "{}: all {} points are in use; remove one first.": "{}: os {} pontos estão em uso; remova um antes.",
    "Preview selection": "Prévia da seleção",
    "show preview (red)": "mostrar prévia (vermelho)",
    "previewing selection": "prévia da seleção",
    "Points with the same subject number are clicks on the same person or object.":
        "Pontos com o mesmo número de sujeito são cliques na mesma pessoa ou objeto.",
    "Leave this area out of the subject, e.g. another person or a prop it grabbed.":
        "Deixa esta área fora do sujeito, por exemplo outra pessoa ou um objeto que ele pegou junto.",
    "Add another click on the last subject, at the plate centre on this frame.":
        "Adiciona outro clique no último sujeito, no centro da plate, neste quadro.",
    "Add a point for another person or object.": "Adiciona um ponto para outra pessoa ou objeto.",
    "Remove all points; Alpha Gen returns to automatic subject selection.":
        "Remove todos os pontos; o Alpha Gen volta à seleção automática do sujeito.",
    "Tints the tracked selection red in this node's output. Turn it off before rendering.":
        "Pinta de vermelho a seleção rastreada na saída deste nó. Desligue antes de renderizar.",
    "No point: Alpha Gen picks the subject itself.<br>"
    "Click several spots of the same person (head, body, prop) for a whole selection; "
    "tick exclude on a point to leave something out. A new subject number selects another person.<br>"
    "Preview runs only SAM over the range (seconds) and tints the selection red.":
        "Sem ponto: o Alpha Gen escolhe o sujeito sozinho.<br>"
        "Clique em vários lugares da mesma pessoa (cabeça, corpo, objeto) para selecioná-la inteira; "
        "marque excluir num ponto para deixar algo de fora. Outro número de sujeito seleciona outra pessoa.<br>"
        "A prévia roda só o SAM no intervalo (segundos) e pinta a seleção de vermelho.",
    "{}: this node predates the selection preview; create a new Alpha Gen node.":
        "{}: este nó é anterior à prévia da seleção; crie um novo nó Alpha Gen.",
    "{}: add a point on the subject first (+ point).": "{}: adicione antes um ponto no sujeito (+ ponto).",
    "selection preview: {} frames in {:.1f}s (red); press Compute when it looks right":
        "prévia da seleção: {} quadros em {:.1f}s (vermelho); clique em Calcular quando estiver certa",
    "SAM prompts as JSON [[frame, x, y, subject, include], ...]; points of one subject are clicks on it.":
        "Pontos do SAM em JSON [[quadro, x, y, sujeito, incluir], ...]; pontos do mesmo sujeito são cliques nele.",
    "only track the SAM selection into --output (subject.alpha), without Alpha Gen":
        "só rastreia a seleção do SAM em --output (subject.alpha), sem o Alpha Gen",
    "--subjects-json must be a JSON list": "--subjects-json precisa ser uma lista JSON",
    "Seed SAM 2.1 with a subject point in plate pixels; repeat for more subjects.":
        "Inicia o SAM 2.1 com um ponto do sujeito em pixels da plate; repita para incluir outros.",
    "Put the generated matte in rgba.alpha or in a node-specific layer.":
        "Coloca o matte gerado em rgba.alpha ou numa camada própria do nó.",
    "Short edge used by Alpha Gen. 544 is the safe starting point on a 24 GB GPU; "
    "a Subject + overlap simulation had a 7.7 GiB backend budget (7.3 GiB free plus 0.44 GiB resident). "
    "A physical 8 GB GPU was not tested; 32 GB of host RAM is a cautious starting point. "
    "Higher settings use substantially more VRAM.":
        "Lado menor usado pelo Alpha Gen. 544 é o ponto de partida seguro numa GPU de 24 GB. "
        "Uma simulação de Subject + sobreposição teve orçamento de 7,7 GiB para o backend (7,3 GiB livres mais 0,44 GiB já residente). "
        "Uma GPU física de 8 GB não foi testada; 32 GB de RAM é um ponto de partida cauteloso. "
        "Valores maiores usam bem mais VRAM.",
    "small VAE tiles": "tiles VAE pequenos",
    "Automatic on physical GPUs up to 8 GB. Enable this on larger GPUs constrained by other apps. "
    "Small tiles can change matte edges; the unchecked 24 GB path keeps the original decode.":
        "Automático em GPUs físicas de até 8 GB. Ative numa placa maior limitada por outros aplicativos. "
        "Tiles pequenos podem alterar as bordas do matte; desmarcado numa placa de 24 GB mantém o decode original.",
    "Fixed seed keeps recomputes reproducible.": "Uma seed fixa faz os recálculos saírem iguais.",
    "Auto reads a connected Read sequence directly. Otherwise, or with Render input, "
    "the Source input is rendered to EXR before Alpha Gen runs.":
        "Auto lê direto uma sequência de um Read conectado. Nos outros casos, ou com Render input, "
        "a entrada Source é renderizada em EXR antes de o Alpha Gen rodar.",
    "EXR sequence written by Compute; empty uses a folder beside the script.":
        "Sequência EXR gravada pelo Calcular; vazio usa uma pasta ao lado do script.",
    "LTX-2.5 Alpha Gen automatically selects the main subject.<br>"
    "Subject points use optional SAM 2.1 setup (~1.2 GB of checkpoints plus a separate CUDA environment).<br>"
    "SAM tracks the chosen subject; without a point, Alpha Gen selects automatically.<br>"
    "The dilated and softened SAM mask only crops the Alpha Gen matte; it cannot recover a missed subject.<br>"
    "Use the generated matte as the alpha of the original plate.<br>"
    "Long shots are processed in overlapping chunks of up to 121 frames with blended boundaries.":
        "O LTX-2.5 Alpha Gen escolhe sozinho o assunto principal.<br>"
        "Os pontos Subject usam o setup opcional do SAM 2.1 (~1,2 GB de checkpoints e ambiente CUDA separado).<br>"
        "O SAM rastreia o sujeito escolhido; sem ponto, o Alpha Gen escolhe automaticamente.<br>"
        "A máscara do SAM, dilatada e suavizada, só recorta o matte do Alpha Gen; ela não recupera um sujeito ignorado.<br>"
        "Use o matte gerado como alpha da plate original.<br>"
        "Planos longos são processados em trechos sobrepostos de até 121 quadros, com fusão nas emendas.",
    # Nuke messages and status
    "{}: connect a source first.": "{}: conecte uma origem primeiro.",
    "{}: last frame is before first frame.": "{}: o último quadro vem antes do primeiro.",
    "subject {} is on frame {}, outside the range {}-{}":
        "o sujeito {} está no quadro {}, fora do intervalo {}-{}",
    "subject {} is outside the plate bounds ({}x{})":
        "o sujeito {} está fora dos limites da plate ({}x{})",
    "{}: {} is already computing.": "{}: {} já está calculando.",
    "{} engine not installed in:\n{}\nRun the tool's installer first.":
        "O motor do {} não está instalado em:\n{}\nRode o instalador da ferramenta primeiro.",
    "{}: runtime check failed: {}": "{}: a verificação falhou: {}",
    "{} is not ready:\n{}": "O {} não está pronto:\n{}",
    "{} failed:\n{}": "O {} falhou:\n{}",
    "{}: rendering input": "{}: renderizando a entrada",
    "{}: rendering the input failed:\n{}": "{}: a renderização da entrada falhou:\n{}",
    "starting engine...": "iniciando o motor...",
    "cancelled": "cancelado",
    "failed: {}": "falhou: {}",
    "cached {} new frames in {:.1f}s": "{} quadros novos no cache em {:.1f}s",
    "computing": "calculando",
    "matting": "gerando matte",
    "{}x{}, {} px short edge, {} chunks": "{}x{}, lado menor {} px, {} trechos",
    "generated {} alpha frames in {:.1f}s": "{} quadros de alpha gerados em {:.1f}s",
    # Alpha Gen engine
    "Alpha Gen: ComfyUI, LTX-2.5 workflow, and adapter are ready.":
        "Alpha Gen: ComfyUI, workflow LTX-2.5 e adaptador prontos.",
    "Use 544p, close other ComfyUI generations, or free GPU memory before retrying.":
        "Use 544p, feche outras gerações do ComfyUI ou libere memória da GPU antes de tentar de novo.",
    "--subject expects an integer FRAME and numeric X Y coordinates":
        "--subject espera um FRAME inteiro e coordenadas X Y numéricas",
    "--subject X and Y must be finite numbers": "--subject X e Y precisam ser números finitos",
    "ComfyUI returned an invalid queue status; refusing to start SAM 2.1":
        "O ComfyUI retornou um estado de fila inválido; o SAM 2.1 não será iniciado",
    "ComfyUI became busy; Alpha Gen stopped before starting another GPU stage":
        "O ComfyUI ficou ocupado; o Alpha Gen parou antes de iniciar outra etapa de GPU",
    "tracking selected subjects with SAM 2.1": "rastreando os sujeitos selecionados com o SAM 2.1",
    "Alpha Gen workflow not found: {}": "Workflow do Alpha Gen não encontrado: {}",
    "Model files are missing (run the Medratools model download):\n{}":
        "Faltam arquivos de modelo (rode o download de modelos do Medratools):\n{}",
    "ComfyUI is not reachable on port {} (start the Medratools backend): {}":
        "O ComfyUI não responde na porta {} (inicie o backend do Medratools): {}",
    "ComfyUI is missing nodes (ComfyUI-LTXVideo or an older ComfyUI?): {}":
        "Faltam nós no ComfyUI (ComfyUI-LTXVideo ausente ou ComfyUI antigo?): {}",
    "ComfyUI rejected the Alpha Gen workflow:\n{}": "O ComfyUI recusou o workflow do Alpha Gen:\n{}",
    "ComfyUI dropped the Alpha Gen job; was the backend restarted?":
        "O ComfyUI descartou o trabalho do Alpha Gen; o backend foi reiniciado?",
    "ComfyUI Alpha Gen failed: {}": "O Alpha Gen falhou no ComfyUI: {}",
    "ComfyUI completed without writing the Alpha Gen matte in {}":
        "O ComfyUI terminou sem gravar o matte do Alpha Gen em {}",
    "Alpha Gen requires a local ComfyUI server because its clips use ComfyUI's input folder":
        "O Alpha Gen precisa de um ComfyUI local, porque os clipes usam a pasta input do ComfyUI",
    "COMFY_PORT must be between 1 and 65535": "COMFY_PORT precisa estar entre 1 e 65535",
    "chunk {}/{}: {}": "trecho {}/{}: {}",
    "waiting for ComfyUI": "aguardando o ComfyUI",
    "loading models (the first run reads ~40 GB)": "carregando modelos (a primeira vez lê ~40 GB)",
    "sampling step {}/{}": "amostragem passo {}/{}",
    "decoding the matte": "decodificando o matte",
    "GPU out of memory. ": "Sem memória na GPU. ",
    "rendering frame {} ({}/{})": "renderizando quadro {} ({}/{})",
    "frame {} ({}/{}), ~{:.0f}s left": "quadro {} ({}/{}), ~{:.0f}s restantes",
    "unknown error": "erro desconhecido",
    # Medratools installer
    "Medratools is not installed yet: run install first": "O Medratools ainda não está instalado: rode o install primeiro",
    "{} is not empty and is not a ComfyUI folder": "{} não está vazia e não é uma pasta do ComfyUI",
    "Registered the Medratools menu in": "Menu do Medratools registrado em",
    "Using the existing ComfyUI in": "Usando o ComfyUI existente em",
    "Warning: custom_nodes/ComfyUI-LTXVideo is missing; Alpha Gen needs it":
        "Aviso: falta custom_nodes/ComfyUI-LTXVideo, que o Alpha Gen precisa",
    "\nNext: download the models, start the backend, then open Nuke > Medratools > Alpha Gen.":
        "\nPróximos passos: baixe os modelos, inicie o backend e abra Nuke > Medratools > Alpha Gen. "
        "Para escolher sujeitos com pontos, rode antes o setup opcional do SAM.",
    "No Python found for {}; start that ComfyUI yourself on port {}":
        "Nenhum Python encontrado para {}; inicie esse ComfyUI você mesmo na porta {}",
    "ComfyUI for Alpha Gen on http://127.0.0.1:{} (Ctrl+C stops it)":
        "ComfyUI do Alpha Gen em http://127.0.0.1:{} (Ctrl+C encerra)",
    "All Alpha Gen models are in": "Todos os modelos do Alpha Gen estão em",
    "The LTX-2.5 weights are licensed under the LTX-2 Community License:":
        "Os pesos do LTX-2.5 seguem a LTX-2 Community License:",
    "Accept the terms with your Hugging Face account first:":
        "Aceite os termos com a sua conta do Hugging Face primeiro:",
    "install the optional SAM 2.1 subject-selection environment and weights":
        "instala o ambiente separado do SAM 2.1 e os pesos opcionais para escolher sujeitos",
    "No RotoTrack engine found beside Medratools": "Motor do RotoTrack não encontrado ao lado do Medratools",
    "SAM 2.1 source/config files are missing from {}; reinstall the package":
        "Os arquivos de código/configuração do SAM 2.1 não estão em {}; reinstale o pacote",
    "Reusing the existing SAM environment at": "Reutilizando o ambiente SAM existente em",
    "SAM 2.1 checkpoint already verified:": "Checkpoint do SAM 2.1 já verificado:",
    "Downloading the official Apache-2.0 SAM 2.1 checkpoint:":
        "Baixando o checkpoint oficial Apache-2.0 do SAM 2.1:",
    "SAM 2.1 checkpoint integrity check failed for {}":
        "A verificação de integridade do checkpoint SAM 2.1 falhou para {}",
    "Verified SAM 2.1 checkpoint:": "Checkpoint SAM 2.1 verificado:",
    "SAM 2.1 subject selection is ready in the separate GPU environment at":
        "A seleção de sujeitos com SAM 2.1 está pronta no ambiente separado de GPU em",
    # Alpha Gen person map
    "No point: Alpha Gen picks the subject itself.<br>"
    "1. Analyze people (once, ~15 s): finds and tracks everyone on this frame; they show faintly.<br>"
    "2. Drag a point onto a person: the selection updates at once, outlined in its colour.<br>"
    "Exclude on a point drops that person. Points off the people (props, someone not found) "
    "are SAM clicks: check them with Preview selection (red).":
        "Sem ponto: o Alpha Gen escolhe o sujeito sozinho.<br>"
        "1. Analisar pessoas (uma vez, ~15 s): acha e rastreia todos neste quadro; aparecem em cor fraca.<br>"
        "2. Arraste um ponto até uma pessoa: a seleção muda na hora, contornada na cor dela.<br>"
        "Excluir num ponto tira aquela pessoa. Pontos fora das pessoas (objetos, alguém não detectado) "
        "são cliques do SAM: confira com Prévia da seleção (vermelho).",
    "Analyze people": "Analisar pessoas",
    "Find the people on the current frame and track them over the range (runs once, then exits).":
        "Acha as pessoas no quadro atual e as rastreia no intervalo (roda uma vez e encerra).",
    "show selection": "mostrar seleção",
    "Shows the selection over this node's output. Turn it off before rendering.":
        "Mostra a seleção sobre a saída do node. Desligue antes de renderizar.",
    "analyzing people": "analisando pessoas",
    "{} people found and tracked in {:.1f}s; drag a point onto the one you want":
        "{} pessoas achadas e rastreadas em {:.1f}s; arraste um ponto até a que você quer",
    "{}: this node predates the person map; create a new Alpha Gen node.":
        "{}: este node é anterior ao mapa de pessoas; crie um node Alpha Gen novo.",
    "{}: every point is on an analyzed person, whose selection already shows; "
    "Preview selection is only needed for points off the people.":
        "{}: todos os pontos estão em pessoas analisadas, cuja seleção já aparece; "
        "a Prévia da seleção só é necessária para pontos fora das pessoas.",
    "Start backend": "Ligar backend",
    "Stop backend": "Desligar backend",
    "Check backend": "Verificar backend",
    "press Check backend": "clique em Verificar backend",
    "backend online": "backend ligado",
    "backend offline": "backend desligado",
    "starting backend...": "ligando backend...",
    "stopping backend...": "desligando backend...",
    "Stop the shared ComfyUI backend? It will only stop if its queue is idle.":
        "Desligar o backend ComfyUI compartilhado? Só será desligado se a fila estiver vazia.",
    "Install comfy-cli to control the backend from Nuke":
        "Instale comfy-cli para controlar o backend pelo Nuke",
    "Backend did not reach the requested state; check comfy logs":
        "O backend não atingiu o estado solicitado; confira comfy logs",
    "The backend port is occupied by an unexpected HTTP service":
        "A porta do backend está ocupada por um serviço HTTP inesperado",
    "{}: no subject is selected; add an include point or reset points for automatic selection.":
        "{}: nenhum sujeito selecionado; adicione um ponto de inclusão ou limpe os pontos para seleção automática.",
    "The person map does not cover {}-{}; press Analyze people again":
        "O mapa de pessoas não cobre {}-{}; clique em Analisar pessoas de novo",
    "Person detection needs ComfyUI's YOLOv10 and ViTPose models in models/detection":
        "A detecção de pessoas precisa dos modelos YOLOv10 e ViTPose do ComfyUI em models/detection",
    "finding and tracking people with SAM 2.1": "achando e rastreando pessoas com o SAM 2.1",
    "track every person found on FRAME into --output (subjects.id map), without Alpha Gen":
        "rastreia cada pessoa achada em FRAME para --output (mapa subjects.id), sem o Alpha Gen",
    "person map EXR pattern written by --analyze-subjects":
        "padrão EXR do mapa de pessoas gravado por --analyze-subjects",
    "comma-separated people of --id-map to keep, e.g. 1,3":
        "pessoas do --id-map a manter, separadas por vírgula, ex.: 1,3",
    "--ids expects comma-separated integers": "--ids espera inteiros separados por vírgula",
}

_CATALOG = PT if language() == "pt" else {}


def tr(text, *args):
    """``text`` in the system language, formatted with ``args``."""
    text = _CATALOG.get(text, text)
    return text.format(*args) if args else text
