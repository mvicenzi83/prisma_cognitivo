#!/usr/bin/env python3
"""
Prisma Cognitivo - Importatore Export ChatGPT
Carica le tue conversazioni ChatGPT come memoria base del Prisma
"""

import json
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    print("❌ ChromaDB richiesto: pip install chromadb")
    HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    print("⚠️ sentence-transformers consigliato: pip install sentence-transformers")
    HAS_EMBEDDINGS = False

class ChatGPTImporter:
    """Importa export ChatGPT nel Prisma Cognitivo"""
    
    def __init__(self, chroma_path="./chroma_journal"):
        self.chroma_path = Path(chroma_path)
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        if HAS_CHROMA:
            self.setup_chroma()
        
        # Pattern per identificare pensieri/domande dell'utente
        self.user_thought_patterns = [
            r"^(sto pensando|mi chiedo|vorrei|ho un dubbio|sono confuso|mi sento)",
            r"^(che ne pensi|cosa faresti|come|perché|quando)",
            r"^(aiutami|consigliami|spiegami)",
            r"(decisione|scelta|dubbio|problema|difficoltà)",
            r"(non so|non riesco|sono incerto|ho paura)"
        ]
    
    def setup_chroma(self):
        """Setup ChromaDB"""
        try:
            print(f"🔗 Connessione ChromaDB: {self.chroma_path}")
            self.client = chromadb.PersistentClient(path=str(self.chroma_path))
            
            # Usa o crea collection
            try:
                self.collection = self.client.get_collection("prisma_thoughts")
                print("✅ Collection 'prisma_thoughts' trovata")
            except:
                self.collection = self.client.create_collection(
                    name="prisma_thoughts",
                    metadata={"description": "Pensieri Prisma + ChatGPT Import"}
                )
                print("🆕 Collection 'prisma_thoughts' creata")
            
            # Setup embeddings
            if HAS_EMBEDDINGS:
                print("🧠 Caricamento modello embeddings...")
                self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                print("✅ Embeddings pronti")
                
        except Exception as e:
            print(f"❌ Errore ChromaDB: {e}")
            self.client = None
    
    def is_meaningful_thought(self, text: str) -> bool:
        """Identifica se un messaggio è un pensiero significativo"""
        if not text or len(text.strip()) < 20:
            return False
        
        text_lower = text.lower().strip()
        
        # Esclude messaggi troppo generici
        generic_phrases = [
            "grazie", "perfetto", "ok", "bene", "sì", "no", 
            "continua", "altro", "stop", "aiuto", "ciao"
        ]
        
        if any(text_lower.startswith(phrase) for phrase in generic_phrases):
            return False
        
        # Cerca pattern di pensieri/riflessioni
        for pattern in self.user_thought_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Se è una domanda articolata
        if "?" in text and len(text) > 30:
            return True
        
        # Se contiene parole chiave riflessive
        reflective_keywords = [
            "riflessione", "pensiero", "idea", "opinione", "consiglio",
            "decisione", "scelta", "futuro", "carriera", "vita", "relazione",
            "progetto", "obiettivo", "sogno", "paura", "speranza",
            "crescita", "cambiamento", "miglioramento"
        ]
        
        if any(keyword in text_lower for keyword in reflective_keywords):
            return True
        
        return False
    
    def extract_meaningful_exchanges(self, conversation: Dict) -> List[Dict]:
        """Estrae scambi significativi da una conversazione"""
        title = conversation.get('title', 'Conversazione senza titolo')
        create_time = conversation.get('create_time', '')
        mapping = conversation.get('mapping', {})
        
        # Estrai messaggi in ordine
        messages = []
        for msg_id, msg_data in mapping.items():
            if not isinstance(msg_data, dict) or not msg_data.get('message'):
                continue
                
            message = msg_data['message']
            if not isinstance(message, dict):
                continue
                
            content = message.get('content', {})
            if isinstance(content, dict) and content.get('parts') and content['parts']:
                text_content = content['parts'][0]
                if text_content and isinstance(text_content, str):
                    role = message.get('author', {}).get('role', 'unknown')
                    create_time_msg = message.get('create_time', '')
                    
                    messages.append({
                        'role': role,
                        'content': text_content.strip(),
                        'timestamp': create_time_msg
                    })
        
        # Trova scambi significativi
        meaningful_exchanges = []
        
        for i in range(len(messages) - 1):
            user_msg = messages[i]
            
            if user_msg['role'] == 'user' and self.is_meaningful_thought(user_msg['content']):
                # Cerca la risposta dell'assistente
                assistant_response = ""
                if i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                    assistant_response = messages[i + 1]['content'][:500]  # Limita lunghezza
                
                meaningful_exchanges.append({
                    'user_thought': user_msg['content'],
                    'assistant_response': assistant_response,
                    'timestamp': user_msg['timestamp'] or create_time,
                    'conversation_title': title,
                    'source': 'chatgpt_import'
                })
        
        return meaningful_exchanges
    
    def parse_chatgpt_export(self, file_path: str) -> List[Dict]:
        """Parsa il file export ChatGPT"""
        try:
            print(f"📂 Caricamento {file_path}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Gestisci diversi formati
            if isinstance(data, list):
                conversations = data
            elif isinstance(data, dict) and 'conversations' in data:
                conversations = data['conversations']
            else:
                print("❌ Formato export non riconosciuto")
                return []
            
            all_exchanges = []
            meaningful_conversations = 0
            
            print(f"🔍 Analizzando {len(conversations)} conversazioni...")
            
            for conv in conversations:
                if not isinstance(conv, dict):
                    continue
                
                exchanges = self.extract_meaningful_exchanges(conv)
                if exchanges:
                    all_exchanges.extend(exchanges)
                    meaningful_conversations += 1
            
            print(f"✅ Trovate {len(all_exchanges)} riflessioni significative")
            print(f"📊 Da {meaningful_conversations} conversazioni utili")
            
            return all_exchanges
            
        except Exception as e:
            print(f"❌ Errore parsing: {e}")
            return []
    
    def create_prisma_documents(self, exchanges: List[Dict]) -> List[Dict]:
        """Converte gli scambi in documenti per il Prisma"""
        documents = []
        
        for exchange in exchanges:
            # Simula una rifrazione del Prisma per ogni pensiero
            user_thought = exchange['user_thought']
            assistant_response = exchange['assistant_response']
            
            # Crea rifrazioni simulate basate sulla risposta ChatGPT
            simulated_refractions = self.simulate_prisma_refractions(user_thought, assistant_response)
            
            # Documento completo
            doc_content = f"PENSIERO ORIGINALE: {user_thought}\n\nRIFRAZIONI:\n"
            
            for lente, rifrazione in simulated_refractions.items():
                doc_content += f"\n{lente}: {rifrazione}\n"
            
            # Metadata ricchi
            timestamp = exchange['timestamp']
            if timestamp:
                try:
                    # Converti timestamp se necessario
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp)
                    else:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    dt = datetime.now()
            else:
                dt = datetime.now()
            
            doc_id = f"chatgpt_import_{dt.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            metadata = {
                "tipo": "pensiero_prisma",
                "fonte": "chatgpt_import",
                "timestamp": dt.isoformat(),
                "conversazione_titolo": exchange['conversation_title'],
                "modello": "chatgpt_import",
                "temperatura": 0.7,
                "num_rifrazioni": len(simulated_refractions),
                "data_creazione": dt.strftime("%Y-%m-%d"),
                "ora_creazione": dt.strftime("%H:%M"),
                "lenti": "Explorer,Connector,Critic,Synthesizer,Visionary,Reflector",
                "pensiero_originale": user_thought[:200]  # Per ricerche
            }
            
            documents.append({
                "id": doc_id,
                "content": doc_content,
                "metadata": metadata
            })
        
        return documents
    
    def simulate_prisma_refractions(self, user_thought: str, assistant_response: str) -> Dict[str, str]:
        """Simula rifrazioni del Prisma basate sulla risposta ChatGPT"""
        
        # Dividi la risposta in sezioni logiche
        response_parts = self.split_response_into_parts(assistant_response)
        
        # Mappa alle lenti del Prisma
        simulated = {
            "🔴 Explorer": self.extract_exploratory_content(response_parts, user_thought),
            "🟠 Connector": self.extract_connecting_content(response_parts, user_thought),
            "🟡 Critic": self.extract_critical_content(response_parts, user_thought),
            "🟢 Synthesizer": self.extract_synthesis_content(response_parts, user_thought),
            "🔵 Visionary": self.extract_visionary_content(response_parts, user_thought),
            "🟣 Reflector": self.extract_reflective_content(response_parts, user_thought)
        }
        
        return simulated
    
    def split_response_into_parts(self, response: str) -> List[str]:
        """Divide la risposta in parti logiche"""
        # Divide per paragrafi, liste, o punti
        parts = []
        
        # Split per paragrafi
        paragraphs = response.split('\n\n')
        for p in paragraphs:
            if p.strip():
                parts.append(p.strip())
        
        # Se non ci sono paragrafi, split per frasi
        if len(parts) <= 1:
            sentences = re.split(r'[.!?]+', response)
            parts = [s.strip() for s in sentences if s.strip()]
        
        return parts[:6]  # Max 6 parti per le 6 lenti
    
    def extract_exploratory_content(self, parts: List[str], thought: str) -> str:
        """Estrae contenuto esplorativo"""
        for part in parts:
            if any(keyword in part.lower() for keyword in [
                'possibilità', 'potresti', 'considera', 'esplora', 'prova', 'diversi', 'alternative'
            ]):
                return part[:200] + "..." if len(part) > 200 else part
        
        return f"Questo pensiero apre diverse possibilità di esplorazione e crescita personale."
    
    def extract_connecting_content(self, parts: List[str], thought: str) -> str:
        """Estrae contenuto di connessione"""
        for part in parts:
            if any(keyword in part.lower() for keyword in [
                'collegato', 'relazione', 'connesso', 'simile', 'come', 'esperienza', 'passato'
            ]):
                return part[:200] + "..." if len(part) > 200 else part
        
        return f"Questo tema si collega a esperienze e conoscenze precedenti nel tuo percorso."
    
    def extract_critical_content(self, parts: List[str], thought: str) -> str:
        """Estrae contenuto critico"""
        for part in parts:
            if any(keyword in part.lower() for keyword in [
                'tuttavia', 'però', 'attenzione', 'considera che', 'importante', 'valuta', 'rischi'
            ]):
                return part[:200] + "..." if len(part) > 200 else part
        
        return f"È importante valutare diversi aspetti e considerare possibili sfide."
    
    def extract_synthesis_content(self, parts: List[str], thought: str) -> str:
        """Estrae contenuto di sintesi"""
        for part in parts:
            if any(keyword in part.lower() for keyword in [
                'riassumendo', 'sintesi', 'quindi', 'conclusione', 'complessivamente', 'organizza'
            ]):
                return part[:200] + "..." if len(part) > 200 else part
        
        return f"Organizzando tutti gli elementi, emerge un quadro strutturato della situazione."
    
    def extract_visionary_content(self, parts: List[str], thought: str) -> str:
        """Estrae contenuto visionario"""
        for part in parts:
            if any(keyword in part.lower() for keyword in [
                'futuro', 'obiettivo', 'visione', 'lungo termine', 'evoluzione', 'crescita', 'sviluppo'
            ]):
                return part[:200] + "..." if len(part) > 200 else part
        
        return f"Guardando al futuro, questo pensiero può evolvere verso nuove direzioni di crescita."
    
    def extract_reflective_content(self, parts: List[str], thought: str) -> str:
        """Estrae contenuto riflessivo"""
        for part in parts:
            if any(keyword in part.lower() for keyword in [
                'riflessione', 'meta', 'processo', 'modo di pensare', 'apprendimento', 'insight'
            ]):
                return part[:200] + "..." if len(part) > 200 else part
        
        return f"Questa riflessione rivela pattern interessanti nel tuo processo di pensiero."
    
    def import_to_chromadb(self, documents: List[Dict]) -> bool:
        """Importa documenti in ChromaDB"""
        if not self.collection:
            print("❌ ChromaDB non disponibile")
            return False
        
        try:
            print(f"💾 Importando {len(documents)} documenti in ChromaDB...")
            
            # Prepara dati per ChromaDB
            ids = [doc["id"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # Batch import
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                
                batch_ids = ids[i:end_idx]
                batch_contents = contents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_contents,
                    metadatas=batch_metadatas
                )
                
                print(f"✅ Importati {end_idx}/{len(documents)} documenti...")
            
            print("🎉 Import completato!")
            return True
            
        except Exception as e:
            print(f"❌ Errore import: {e}")
            return False
    
    def show_import_stats(self):
        """Mostra statistiche post-import"""
        if not self.collection:
            return
        
        try:
            total_count = self.collection.count()
            
            # Conta documenti importati
            import_results = self.collection.get(
                where={"fonte": "chatgpt_import"},
                include=["metadatas"]
            )
            
            import_count = len(import_results['metadatas']) if import_results['metadatas'] else 0
            
            print(f"\n📊 STATISTICHE POST-IMPORT")
            print("─" * 40)
            print(f"📚 Documenti totali in ChromaDB: {total_count}")
            print(f"📥 Documenti importati da ChatGPT: {import_count}")
            print(f"💡 Nuova memoria disponibile per il Prisma!")
            
            if import_results['metadatas']:
                # Mostra range temporale
                timestamps = [m.get('timestamp') for m in import_results['metadatas'] if m.get('timestamp')]
                if timestamps:
                    timestamps.sort()
                    primo = datetime.fromisoformat(timestamps[0]).strftime("%d/%m/%Y")
                    ultimo = datetime.fromisoformat(timestamps[-1]).strftime("%d/%m/%Y")
                    print(f"📅 Range temporale: {primo} - {ultimo}")
            
        except Exception as e:
            print(f"❌ Errore statistiche: {e}")


def main():
    """Funzione principale per l'import"""
    print("🌈 PRISMA COGNITIVO - IMPORTATORE CHATGPT")
    print("=" * 60)
    print("📥 Importa le tue conversazioni ChatGPT come memoria base")
    print("=" * 60)
    
    # Chiedi file ChatGPT
    print("\n📂 Path del file export ChatGPT (.json):")
    file_path = input("📁 ").strip().strip('"').strip("'")
    
    if not file_path or not Path(file_path).exists():
        print("❌ File non trovato!")
        print("💡 Esporta le tue conversazioni da ChatGPT in formato JSON")
        return
    
    # Chiedi path ChromaDB
    print(f"\n🗂️ Path ChromaDB (Enter per default './chroma_journal'):")
    chroma_path = input("📁 ").strip()
    if not chroma_path:
        chroma_path = "./chroma_journal"
    
    # Inizializza importer
    importer = ChatGPTImporter(chroma_path)
    
    if not importer.collection:
        print("❌ Impossibile inizializzare ChromaDB")
        return
    
    # Mostra stato iniziale
    initial_count = importer.collection.count()
    print(f"\n📊 Documenti attuali in ChromaDB: {initial_count}")
    
    # Conferma import
    proceed = input(f"\n🚀 Procedere con l'import da {Path(file_path).name}? (s/n): ").lower()
    if proceed not in ['s', 'si', 'y', 'yes']:
        print("❌ Import annullato")
        return
    
    # Parsing
    exchanges = importer.parse_chatgpt_export(file_path)
    
    if not exchanges:
        print("❌ Nessuna conversazione significativa trovata")
        return
    
    print(f"\n🔍 Trovate {len(exchanges)} riflessioni da importare")
    
    # Chiedi conferma finale
    final_confirm = input("📥 Confermi l'import in ChromaDB? (s/n): ").lower()
    if final_confirm not in ['s', 'si', 'y', 'yes']:
        print("❌ Import annullato")
        return
    
    # Crea documenti Prisma
    print("🔄 Conversione in formato Prisma...")
    documents = importer.create_prisma_documents(exchanges)
    
    # Import in ChromaDB
    success = importer.import_to_chromadb(documents)
    
    if success:
        # Mostra statistiche
        importer.show_import_stats()
        print(f"\n🎉 IMPORT COMPLETATO!")
        print("✨ Il tuo Prisma Cognitivo ora ha una memoria ricchissima!")
        print("🚀 Lancia il Prisma per vedere le nuove connessioni!")
    else:
        print("❌ Import fallito")

if __name__ == "__main__":
    main()