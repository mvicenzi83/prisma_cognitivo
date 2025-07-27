#!/usr/bin/env python3
"""
Journal AI 2.0 - RAG con Monitoring Automatico
Elabora export ChatGPT e monitora automaticamente nuovi file
"""

import json
import os
import time
import threading
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set
import argparse
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Installazione dipendenze necessarie
# pip install langchain langchain-community chromadb ollama sentence-transformers watchdog

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama
    from langchain.schema import Document
    from langchain.chains import RetrievalQA
except ImportError:
    print("❌ Dipendenze mancanti. Installa con:")
    print("pip install langchain langchain-community chromadb ollama sentence-transformers watchdog")
    exit(1)

class JournalFileHandler(FileSystemEventHandler):
    """Handler per monitorare nuovi file JSON"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.processed_files = set()
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            self.rag_system.logger.info(f"📁 Nuovo file rilevato: {event.src_path}")
            time.sleep(2)  # Aspetta che il file sia completamente scritto
            self.rag_system.process_new_file(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            file_path = event.src_path
            if file_path not in self.processed_files:
                self.logger.info(f"File modificato: {file_path}")
                time.sleep(1)
                self.rag_system.process_new_file(file_path)

class JournalRAG:
    def __init__(self, ollama_model="gemma:latest", data_dir="./data", persist_dir="./chroma_journal"):
        self.ollama_model = ollama_model
        self.data_dir = Path(data_dir)
        self.persist_dir = persist_dir
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.processed_files = set()
        self.observer = None
        
        # Setup logging
        self.setup_logging()
        
        # Crea directory necessarie
        self.data_dir.mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Configura il logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./logs/journal_rag.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_file_hash(self, file_path: str) -> str:
        """Calcola hash del file per rilevare modifiche"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_processed_files(self):
        """Carica lista file già processati"""
        processed_file = Path("./logs/processed_files.json")
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                data = json.load(f)
                self.processed_files = set(data.get('files', []))
    
    def save_processed_files(self):
        """Salva lista file processati"""
        processed_file = Path("./logs/processed_files.json")
        with open(processed_file, 'w') as f:
            json.dump({'files': list(self.processed_files), 'updated': datetime.now().isoformat()}, f)
    
    def parse_chatgpt_export(self, file_path: str) -> List[Dict]:
        """Elabora il file JSON esportato da ChatGPT"""
        self.logger.info(f"Caricamento {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Errore lettura file {file_path}: {e}")
            return []
        
        conversations = []
        
        # Gestisci diversi formati di export ChatGPT
        if isinstance(data, list):
            conversations_data = data
        elif isinstance(data, dict) and 'conversations' in data:
            conversations_data = data['conversations']
        else:
            self.logger.error(f"Formato file non riconosciuto: {file_path}")
            return []
        
        for conversation in conversations_data:
            if not isinstance(conversation, dict):
                continue
                
            title = conversation.get('title', 'Conversazione senza titolo')
            create_time = conversation.get('create_time', '')
            
            # Estrai i messaggi
            mapping = conversation.get('mapping', {})
            messages = []
            
            for msg_id, msg_data in mapping.items():
                if not isinstance(msg_data, dict) or not msg_data.get('message'):
                    continue
                    
                message = msg_data['message']
                if not isinstance(message, dict):
                    continue
                    
                content = message.get('content', {})
                
                if isinstance(content, dict) and content.get('parts') and content['parts']:
                    # Gestisci il caso in cui parts[0] potrebbe essere None o non stringa
                    text_content = content['parts'][0]
                    if text_content and isinstance(text_content, str) and text_content.strip():
                        role = message.get('author', {}).get('role', 'unknown')
                        
                        messages.append({
                            'role': role,
                            'content': text_content,
                            'timestamp': message.get('create_time', '')
                        })
            
            if messages:
                conversations.append({
                    'title': title,
                    'create_time': create_time,
                    'messages': messages,
                    'source_file': os.path.basename(file_path)
                })
        
        self.logger.info(f"Trovate {len(conversations)} conversazioni in {file_path}")
        return conversations
    
    def create_documents(self, conversations: List[Dict]) -> List[Document]:
        """Converte conversazioni in documenti per RAG"""
        documents = []
        
        for conv in conversations:
            # Documento conversazione completa
            full_text = f"Titolo: {conv['title']}\n\n"
            
            for msg in conv['messages']:
                role_name = "Tu" if msg['role'] == 'user' else "AI"
                full_text += f"{role_name}: {msg['content']}\n\n"
            
            metadata = {
                'title': conv['title'],
                'create_time': conv['create_time'],
                'message_count': len(conv['messages']),
                'type': 'conversation',
                'source_file': conv.get('source_file', 'unknown'),
                'doc_id': f"conv_{hashlib.md5(full_text.encode()).hexdigest()[:8]}"
            }
            
            documents.append(Document(page_content=full_text, metadata=metadata))
            
            # Documenti per singoli scambi
            for i in range(0, len(conv['messages'])-1, 2):
                if i+1 < len(conv['messages']):
                    user_msg = conv['messages'][i]
                    ai_msg = conv['messages'][i+1]
                    
                    if user_msg['role'] == 'user' and ai_msg['role'] == 'assistant':
                        exchange_text = f"Domanda: {user_msg['content']}\n\nRisposta: {ai_msg['content']}"
                        
                        exchange_metadata = {
                            'title': conv['title'],
                            'create_time': conv['create_time'],
                            'type': 'exchange',
                            'user_question': user_msg['content'][:100] + "...",
                            'source_file': conv.get('source_file', 'unknown'),
                            'doc_id': f"exch_{hashlib.md5(exchange_text.encode()).hexdigest()[:8]}"
                        }
                        
                        documents.append(Document(page_content=exchange_text, metadata=exchange_metadata))
        
        return documents
    
    def setup_embeddings(self):
        """Configura gli embeddings"""
        if self.embeddings is None:
            self.logger.info("Configurazione embeddings con Ollama...")
            try:
                self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
                # Test dell'embedding
                test_embed = self.embeddings.embed_query("test")
                self.logger.info("Embeddings nomic-embed-text configurati")
            except Exception as e:
                self.logger.warning(f"nomic-embed-text non disponibile: {e}")
                self.logger.info("Uso gemma per embeddings...")
                self.embeddings = OllamaEmbeddings(model=self.ollama_model)
    
    def setup_vectorstore(self, documents: List[Document], append_mode=False):
        """Configura o aggiorna il vector store"""
        self.setup_embeddings()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        self.logger.info(f"📝 Creati {len(split_docs)} chunks")
        
        if append_mode and os.path.exists(self.persist_dir):
            # Carica vectorstore esistente e aggiungi documenti
            self.logger.info("📚 Caricamento vectorstore esistente...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            
            # Filtra documenti duplicati basandosi su doc_id
            existing_ids = set()
            try:
                existing_docs = self.vectorstore.get()
                for metadata in existing_docs['metadatas']:
                    if metadata and 'doc_id' in metadata:
                        existing_ids.add(metadata['doc_id'])
            except:
                pass
            
            new_docs = []
            for doc in split_docs:
                doc_id = doc.metadata.get('doc_id')
                if doc_id and doc_id not in existing_ids:
                    new_docs.append(doc)
            
            if new_docs:
                self.logger.info(f"➕ Aggiunta di {len(new_docs)} nuovi documenti...")
                self.vectorstore.add_documents(new_docs)
                self.vectorstore.persist()
                self.logger.info("✅ Vectorstore aggiornato")
            else:
                self.logger.info("ℹ️ Nessun nuovo documento da aggiungere")
                
        else:
            # Crea nuovo vectorstore
            self.logger.info("💾 Creazione vectorstore...")
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            self.logger.info(f"✅ Vectorstore creato in {self.persist_dir}")
    
    def setup_qa_chain(self):
        """Configura la catena QA"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore non inizializzato!")
            
        self.logger.info("🤖 Configurazione modello Ollama...")
        
        llm = Ollama(model=self.ollama_model, temperature=0.7)
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        self.logger.info("✅ Catena QA configurata!")
    
    def process_new_file(self, file_path: str):
        """Processa un nuovo file JSON"""
        try:
            file_hash = self.get_file_hash(file_path)
            file_key = f"{file_path}:{file_hash}"
            
            if file_key in self.processed_files:
                self.logger.info(f"⏭️ File già processato: {file_path}")
                return
            
            self.logger.info(f"🔄 Processamento nuovo file: {file_path}")
            
            conversations = self.parse_chatgpt_export(file_path)
            if conversations:
                documents = self.create_documents(conversations)
                self.setup_vectorstore(documents, append_mode=True)
                
                # Ricarica QA chain se necessario
                if self.qa_chain:
                    self.setup_qa_chain()
                    
                self.processed_files.add(file_key)
                self.save_processed_files()
                
                self.logger.info(f"✅ File processato con successo: {file_path}")
            else:
                self.logger.warning(f"⚠️ Nessuna conversazione trovata in: {file_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Errore processamento {file_path}: {e}")
    
    def start_monitoring(self):
        """Avvia il monitoring automatico"""
        self.logger.info(f"👁️ Avvio monitoring directory: {self.data_dir}")
        
        event_handler = JournalFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.data_dir), recursive=False)
        self.observer.start()
        
        self.logger.info("✅ Monitoring attivo!")
    
    def stop_monitoring(self):
        """Ferma il monitoring"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.logger.info("🛑 Monitoring fermato")
    
    def process_existing_files(self):
        """Processa tutti i file JSON esistenti nella directory"""
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            self.logger.warning(f"⚠️ Nessun file JSON trovato in {self.data_dir}")
            return
        
        self.logger.info(f"Trovati {len(json_files)} file JSON")
        
        all_conversations = []
        for file_path in json_files:
            conversations = self.parse_chatgpt_export(str(file_path))
            all_conversations.extend(conversations)
            
            # Marca come processato
            file_hash = self.get_file_hash(str(file_path))
            self.processed_files.add(f"{file_path}:{file_hash}")
        
        if all_conversations:
            documents = self.create_documents(all_conversations)
            self.setup_vectorstore(documents, append_mode=False)
            self.setup_qa_chain()
            self.save_processed_files()
    
    def query(self, question: str) -> Dict[str, Any]:
        """Interroga il sistema RAG"""
        if not self.qa_chain:
            raise ValueError("Sistema RAG non inizializzato!")
        
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]],
            "source_texts": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
        }

def interactive_mode(rag: JournalRAG):
    """Modalità interattiva per query"""
    print("\n🎯 Modalità interattiva attiva!")
    print("🔍 Fai le tue domande (o 'quit' per uscire, 'stats' per statistiche)")

    # 👉 AGGIUNGI QUESTO LOG QUI SUBITO
    rag.logger.info("🟢 Entrato in modalità interattiva")  
    print("🟢 In attesa delle tue domande...")  # output visivo immediato
    
    while True:
        try:
            question = input("\n❓ Domanda: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            elif question.lower() == 'stats':
                if rag.vectorstore:
                    try:
                        count = rag.vectorstore._collection.count()
                        print(f"📊 Documenti nel database: {count}")
                        print(f"📁 File processati: {len(rag.processed_files)}")
                    except:
                        print("📊 Statistiche non disponibili")
                continue
            
            if question:
                result = rag.query(question)
                print(f"\n🤖 {result['answer']}")
                
                if result['sources']:
                    print(f"\n📚 Fonti ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'][:3], 1):
                        title = source.get('title', 'Senza titolo')[:50]
                        source_file = source.get('source_file', 'N/A')
                        print(f"   {i}. {title}... (da {source_file})")
                        
        except KeyboardInterrupt:
            print("\n👋 Uscita...")
            break
        except Exception as e:
            print(f"❌ Errore: {e}")


def main():
    parser = argparse.ArgumentParser(description="Journal AI 2.0 - RAG con Monitoring")
    parser.add_argument("--data-dir", default="./data", help="Directory dei file JSON")
    parser.add_argument("--model", default="gemma:latest", help="Modello Ollama")
    parser.add_argument("--no-monitor", action="store_true", help="Disabilita monitoring automatico")
    
    args = parser.parse_args()
    
    print("🚀 Journal AI 2.0 - RAG con Monitoring Automatico")
    print("=" * 60)
    
    # Inizializza sistema RAG
    rag = JournalRAG(ollama_model=args.model, data_dir=args.data_dir)
    
    try:
        # Carica file già processati
        rag.load_processed_files()
        
        # Processa file esistenti
        rag.process_existing_files()
        
        # Avvia monitoring se richiesto
        if not args.no_monitor:
            rag.start_monitoring()
            
            print(f"\n📁 Monitoring attivo su: {args.data_dir}")
            print("💡 Aggiungi nuovi file JSON nella directory per aggiornarli automaticamente!")
        
        # Modalità interattiva
        interactive_mode(rag)
        
    except KeyboardInterrupt:
        print("\n🛑 Interruzione ricevuta...")
    except Exception as e:
        print(f"❌ Errore: {e}")
    finally:
        rag.stop_monitoring()
        print("👋 Arrivederci!")

if __name__ == "__main__":
    main()
