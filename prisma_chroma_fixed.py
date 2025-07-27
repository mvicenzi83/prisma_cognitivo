#!/usr/bin/env python3
"""
Prisma Cognitivo - Versione ChromaDB Fixed
Fix per compatibilità ChromaDB e gestione errori Ollama
"""

import asyncio
import aiohttp
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    print("⚠️ ChromaDB non trovato. Installalo con: pip install chromadb")
    HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    print("⚠️ sentence-transformers non trovato. Installalo con: pip install sentence-transformers")
    HAS_EMBEDDINGS = False

class ChromaPrismaMemoryFixed:
    """Memoria del Prisma con fix per ChromaDB"""
    
    def __init__(self, chroma_path="./chroma_journal"):
        self.chroma_path = Path(chroma_path)
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        if HAS_CHROMA:
            self.setup_chroma()
        else:
            print("❌ ChromaDB non disponibile - modalità senza memoria")
    
    def setup_chroma(self):
        """Inizializza connessione ChromaDB"""
        try:
            print(f"🔗 Connessione a ChromaDB: {self.chroma_path}")
            
            self.client = chromadb.PersistentClient(path=str(self.chroma_path))
            
            try:
                self.collection = self.client.get_collection("prisma_thoughts")
                print(f"✅ Collection 'prisma_thoughts' trovata!")
            except:
                print("🆕 Creo nuova collection 'prisma_thoughts'")
                self.collection = self.client.create_collection(
                    name="prisma_thoughts",
                    metadata={"description": "Pensieri del Prisma Cognitivo"}
                )
            
            if HAS_EMBEDDINGS:
                print("🧠 Caricamento modello embeddings...")
                self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                print("✅ Embeddings pronti")
            
            count = self.collection.count()
            print(f"📚 Documenti in memoria: {count}")
            
        except Exception as e:
            print(f"❌ Errore ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def salva_pensiero(self, contenuto: str, rifrazioni: Dict, sessione_id: str, modello: str, temperatura: float):
        """Salva pensiero in ChromaDB"""
        if not self.collection:
            return
        
        try:
            doc_id = f"pensiero_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            metadata = {
                "tipo": "pensiero_prisma",
                "timestamp": datetime.now().isoformat(),
                "sessione_id": sessione_id,
                "modello": modello,
                "temperatura": temperatura,
                "num_rifrazioni": len(rifrazioni),
                "data_creazione": datetime.now().strftime("%Y-%m-%d"),
                "ora_creazione": datetime.now().strftime("%H:%M")
            }
            
            lenti_usate = list(rifrazioni.keys())
            metadata["lenti"] = ",".join([lente.split()[1] for lente in lenti_usate])
            
            documento_completo = f"""PENSIERO ORIGINALE: {contenuto}

RIFRAZIONI:
"""
            
            for lente, dati in rifrazioni.items():
                documento_completo += f"\n{lente}: {dati.get('output', '')}\n"
            
            self.collection.add(
                documents=[documento_completo],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            print(f"💾 Pensiero salvato in ChromaDB (ID: {doc_id[:16]}...)")
            
        except Exception as e:
            print(f"❌ Errore salvataggio ChromaDB: {e}")
    
    def trova_pensieri_simili(self, testo: str, limite: int = 3) -> List[Dict]:
        """Trova pensieri simili usando ChromaDB (versione fixed)"""
        if not self.collection:
            return []
        
        try:
            risultati = self.collection.query(
                query_texts=[testo],
                n_results=limite,
                where={"tipo": "pensiero_prisma"}
            )
            
            pensieri_simili = []
            
            if risultati['documents'] and risultati['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    risultati['documents'][0],
                    risultati['metadatas'][0], 
                    risultati['distances'][0]
                )):
                    pensiero = doc.split("RIFRAZIONI:")[0].replace("PENSIERO ORIGINALE: ", "").strip()
                    
                    pensieri_simili.append({
                        'contenuto': pensiero,
                        'timestamp': metadata.get('timestamp', ''),
                        'similarita': 1 - distance,
                        'modello': metadata.get('modello', ''),
                        'lenti': metadata.get('lenti', '')
                    })
            
            return pensieri_simili
            
        except Exception as e:
            print(f"❌ Errore ricerca ChromaDB: {e}")
            return []
    
    def get_statistiche(self) -> Dict:
        """Ottieni statistiche dalla memoria"""
        if not self.collection:
            return {"pensieri_totali": 0, "errore": "ChromaDB non disponibile"}
        
        try:
            count = self.collection.count()
            
            # Versione semplificata senza filtri temporali complessi
            try:
                risultati = self.collection.get(limit=1000, include=["metadatas"])
                if risultati['metadatas']:
                    timestamps = [m.get('timestamp') for m in risultati['metadatas'] if m.get('timestamp')]
                    if timestamps:
                        timestamps.sort()
                        primo = timestamps[0]
                        ultimo = timestamps[-1]
                    else:
                        primo = ultimo = None
                else:
                    primo = ultimo = None
            except:
                primo = ultimo = None
            
            return {
                "pensieri_totali": count,
                "primo_pensiero": primo,
                "ultimo_pensiero": ultimo,
                "database_path": str(self.chroma_path)
            }
            
        except Exception as e:
            return {"pensieri_totali": 0, "errore": str(e)}
    
    def get_pensieri_recenti(self, giorni: int = 7) -> List[Dict]:
        """Ottieni pensieri recenti (versione fixed)"""
        if not self.collection:
            return []
        
        try:
            # Semplificato: prendi tutti e filtra manualmente
            risultati = self.collection.get(
                where={"tipo": "pensiero_prisma"},
                include=["documents", "metadatas"],
                limit=100
            )
            
            pensieri_recenti = []
            data_limite = datetime.now() - timedelta(days=giorni)
            
            if risultati['documents']:
                for doc, metadata in zip(risultati['documents'], risultati['metadatas']):
                    timestamp_str = metadata.get('timestamp', '')
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if timestamp >= data_limite:
                                pensiero = doc.split("RIFRAZIONI:")[0].replace("PENSIERO ORIGINALE: ", "").strip()
                                pensieri_recenti.append({
                                    'contenuto': pensiero,
                                    'timestamp': timestamp_str,
                                    'modello': metadata.get('modello', ''),
                                    'sessione_id': metadata.get('sessione_id', '')
                                })
                        except:
                            continue
            
            pensieri_recenti.sort(key=lambda x: x['timestamp'], reverse=True)
            return pensieri_recenti[:20]
            
        except Exception as e:
            print(f"❌ Errore ricerca recenti (fixed): {e}")
            return []
    
    def cerca_per_tema(self, tema: str, limite: int = 5) -> List[Dict]:
        """Cerca pensieri per tema specifico"""
        if not self.collection:
            return []
        
        try:
            risultati = self.collection.query(
                query_texts=[tema],
                n_results=limite,
                where={"tipo": "pensiero_prisma"}
            )
            
            pensieri_tema = []
            
            if risultati['documents'] and risultati['documents'][0]:
                for doc, metadata, distance in zip(
                    risultati['documents'][0],
                    risultati['metadatas'][0],
                    risultati['distances'][0]
                ):
                    pensiero = doc.split("RIFRAZIONI:")[0].replace("PENSIERO ORIGINALE: ", "").strip()
                    
                    pensieri_tema.append({
                        'contenuto': pensiero,
                        'timestamp': metadata.get('timestamp', ''),
                        'relevanza': 1 - distance,
                        'lenti_usate': metadata.get('lenti', '')
                    })
            
            return pensieri_tema
            
        except Exception as e:
            print(f"❌ Errore ricerca tema: {e}")
            return []


class PrismaCognitivoChromaFixed:
    """Prisma Cognitivo con ChromaDB e gestione errori migliorata"""
    
    def __init__(self, ollama_host="http://localhost:11434", model="mistral:latest", chroma_path="./chroma_journal"):
        self.ollama_host = ollama_host
        self.model = model
        self.temperature = 0.7
        self.timeout = 60
        self.session_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.memoria = ChromaPrismaMemoryFixed(chroma_path)
        
        self.encouragements = [
            "Ottima riflessione! 🌟",
            "Insights molto profondi! 💎", 
            "Che bella prospettiva! ✨",
            "Riflessione illuminante! 🔮",
            "La tua memoria si arricchisce! 📚",
            "Vedo pattern interessanti! 🔗",
            "Pensiero davvero evolutivo! 🧠",
            "ChromaDB sta imparando da te! 🚀"
        ]
        
        self.cognitive_lenses = {
            "🔴 Explorer": {
                "emoji": "🔴",
                "name": "Explorer",
                "color": "red",
                "prompt": """Sei un ESPLORATORE COGNITIVO esperto di lingua italiana. Il tuo ruolo è esplorare liberamente l'idea presentata, aprire nuove possibilità e identificare potenziali nascosti.

IMPORTANTE: Rispondi sempre in italiano corretto e fluente.

MEMORIA DALLA TUA CRONOLOGIA: {memoria_contesto}

APPROCCIO:
- Esplora l'idea da angolazioni creative, considerando anche i tuoi pattern passati
- Identifica potenziali nascosti e opportunità, collegandoti alla tua storia cognitiva
- Fai domande che aprono nuove direzioni basandoti sulla tua evoluzione
- Sii curioso, aperto e ispirante

Pensiero attuale: {input_text}

Rispondi in italiano perfetto con 3-4 frasi ricche di spunti creativi e connessioni con il tuo passato:"""
            },
            "🟠 Connector": {
                "emoji": "🟠", 
                "name": "Connector",
                "color": "orange",
                "prompt": """Sei un CONNETTORE COGNITIVO esperto di lingua italiana. Il tuo ruolo è trovare collegamenti profondi tra l'idea attuale e la tua ricca storia di riflessioni.

IMPORTANTE: Rispondi sempre in italiano corretto e fluente.

TUA MEMORIA CRONOLOGICA: {memoria_contesto}
ESPLORAZIONE PRECEDENTE: {previous_output}

APPROCCIO:
- Cerca pattern significativi tra questo pensiero e la tua cronologia personale
- Collega a temi ricorrenti nella tua crescita cognitiva
- Identifica l'evoluzione del tuo modo di pensare nel tempo
- Costruisci ponti tra diverse fasi della tua vita riflessiva

Pensiero attuale: {original_input}

Rispondi in italiano perfetto con 3-4 frasi che rivelano connessioni profonde con la tua storia:"""
            },
            "🟡 Critic": {
                "emoji": "🟡",
                "name": "Critic", 
                "color": "yellow",
                "prompt": """Sei un CRITICO COSTRUTTIVO esperto di lingua italiana. Il tuo ruolo è analizzare l'idea con rigore ma sempre in modo edificante, basandoti anche sulla tua esperienza accumulata.

IMPORTANTE: Rispondi sempre in italiano corretto e fluente.

TUA CRONOLOGIA COGNITIVA: {memoria_contesto}
SVILUPPO PRECEDENTE: {previous_output}

APPROCCIO:
- Identifica con gentilezza assunzioni, anche basandoti sui tuoi pattern passati
- Evidenzia aspetti che meritano riflessione, considerando la tua evoluzione
- Proponi domande provocatorie ma costruttive, informate dalla tua storia
- Suggerisci miglioramenti basati sulla tua esperienza cognitiva

Pensiero attuale: {original_input}

Rispondi in italiano perfetto con 3-4 frasi di analisi critica ma sempre costruttiva e informata:"""
            },
            "🟢 Synthesizer": {
                "emoji": "🟢",
                "name": "Synthesizer",
                "color": "green", 
                "prompt": """Sei un SINTETIZZATORE COGNITIVO esperto di lingua italiana. Il tuo ruolo è organizzare tutto ciò che è emerso in una forma chiara, coerente e utile, integrando la tua memoria storica.

IMPORTANTE: Rispondi sempre in italiano corretto e fluente.

TUA MEMORIA PERSONALE: {memoria_contesto}
SVILUPPO COMPLETO: {previous_output}

APPROCCIO:
- Organizza le idee includendo i pattern della tua cronologia
- Trova il filo conduttore che unisce presente e passato nella tua evoluzione
- Crea sintesi che illuminano la tua crescita cognitiva nel tempo
- Proponi framework che tengono conto della tua storia personale

Pensiero attuale: {original_input}

Rispondi in italiano perfetto con 4-5 frasi che sintetizzano tutto in modo evolutivo e personale:"""
            },
            "🔵 Visionary": {
                "emoji": "🔵",
                "name": "Visionary",
                "color": "blue",
                "prompt": """Sei un VISIONARIO COGNITIVO esperto di lingua italiana. Il tuo ruolo è proiettare l'idea nel futuro con ottimismo, basandoti sulla traiettoria della tua crescita personale.

IMPORTANTE: Rispondi sempre in italiano corretto e fluente.

TUA TRAIETTORIA STORICA: {memoria_contesto}
RIFLESSIONE COMPLETA: {previous_output}

APPROCCIO:
- Proietta nel futuro considerando la tua evoluzione personale documentata
- Immagina scenari basati sui pattern di crescita della tua storia
- Considera implicazioni che si allineano con la tua direzione di sviluppo
- Visualizza trasformazioni coerenti con il tuo percorso cognitivo

Pensiero attuale: {original_input}

Rispondi in italiano perfetto con 3-4 frasi che dipingono un futuro coerente con la tua crescita:"""
            },
            "🟣 Reflector": {
                "emoji": "🟣",
                "name": "Reflector",
                "color": "purple",
                "prompt": """Sei un RIFLETTORE METACOGNITIVO esperto di lingua italiana. Il tuo ruolo è riflettere sul processo di pensiero stesso, considerando l'intera tua cronologia cognitiva.

IMPORTANTE: Rispondi sempre in italiano corretto e fluente.

INTERA TUA STORIA COGNITIVA: {memoria_contesto}
VIAGGIO COMPLETO ATTUALE: {previous_output}

APPROCCIO:
- Fai meta-riflessione sulla tua evoluzione cognitiva nel tempo
- Identifica pattern di crescita e apprendimento dalla tua cronologia
- Estrai insight su come il tuo pensiero si è sviluppato
- Offri suggerimenti basati sulla tua storia personale di riflessione

Pensiero attuale: {original_input}

Rispondi in italiano perfetto con 4-5 frasi di meta-riflessione saggia e evolutiva basata sulla tua esperienza:"""
            }
        }

    async def call_ollama(self, prompt: str) -> str:
        """Chiamata asincrona a Ollama con gestione errori migliorata"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "num_ctx": 8192,
                    "repeat_penalty": 1.1
                }
            }
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.ollama_host}/api/generate", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("response", "").strip()
                        if result:
                            return self.clean_response(result)
                        else:
                            return "Il modello ha restituito una risposta vuota. Prova a riformulare il pensiero."
                    elif response.status == 404:
                        return f"❌ Modello '{self.model}' non trovato. Scarica con: ollama pull {self.model}"
                    else:
                        error_text = await response.text()
                        return f"❌ Errore HTTP {response.status}: {error_text}"
        except aiohttp.ClientConnectorError:
            return "❌ Impossibile connettersi a Ollama. Verifica che sia avviato con: ollama serve"
        except asyncio.TimeoutError:
            return f"❌ Timeout dopo {self.timeout}s. Il modello potrebbe essere lento o sovraccarico."
        except Exception as e:
            return f"❌ Errore imprevisto: {str(e)}"
    
    def clean_response(self, text: str) -> str:
        """Pulisce la risposta"""
        text = text.replace("Risposta:", "").strip()
        text = text.replace("Risposta (", "").strip()
        
        if text and not text.endswith(('.', '!', '?')):
            text += "."
            
        return text

    async def test_connection(self) -> bool:
        """Testa connessione Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_host}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return response.status == 200
        except:
            return False

    async def check_model_availability(self, model_name: str) -> bool:
        """Verifica se il modello è disponibile"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_host}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        return model_name in models
                    return False
        except:
            return False

    def print_header(self, input_text: str):
        """Header con info ChromaDB"""
        print("\n" + "=" * 70)
        print("🌈 PRISMA COGNITIVO + CHROMADB (FIXED)")
        print("=" * 70)
        print(f"💭 Pensiero: \"{input_text}\"")
        print(f"🤖 Modello: {self.model}")
        print(f"🌡️  Temperatura: {self.temperature}")
        
        stats = self.memoria.get_statistiche()
        if "errore" not in stats:
            print(f"📚 ChromaDB: {stats['pensieri_totali']} pensieri memorizzati")
            print(f"🗂️  Database: {stats.get('database_path', '')}")
        else:
            print(f"⚠️  ChromaDB: {stats['errore']}")
        print("=" * 70)

    def prepara_contesto_memoria(self, input_text: str) -> str:
        """Prepara contesto dalla memoria ChromaDB"""
        if not self.memoria.collection:
            return "ChromaDB non disponibile - prima sessione senza memoria storica."
        
        pensieri_simili = self.memoria.trova_pensieri_simili(input_text, limite=3)
        pensieri_recenti = self.memoria.get_pensieri_recenti(giorni=30)
        
        contesto_parts = []
        
        if pensieri_simili:
            contesto_parts.append("PENSIERI SIMILI DALLA TUA CRONOLOGIA:")
            for p in pensieri_simili:
                if p['timestamp']:
                    try:
                        timestamp = datetime.fromisoformat(p['timestamp']).strftime("%d/%m/%Y")
                    except:
                        timestamp = "Data sconosciuta"
                else:
                    timestamp = "Data sconosciuta"
                
                similarita_pct = int(p['similarita'] * 100)
                contenuto = p['contenuto'][:100] + "..." if len(p['contenuto']) > 100 else p['contenuto']
                contesto_parts.append(f"- {timestamp} ({similarita_pct}% simile): {contenuto}")
        
        if pensieri_recenti:
            contesto_parts.append("\nTUOI TEMI RECENTI (ultimi 30 giorni):")
            for p in pensieri_recenti[:3]:
                try:
                    timestamp = datetime.fromisoformat(p['timestamp']).strftime("%d/%m")
                except:
                    timestamp = "Data sconosciuta"
                contenuto = p['contenuto'][:80] + "..." if len(p['contenuto']) > 80 else p['contenuto']
                contesto_parts.append(f"- {timestamp}: {contenuto}")
        
        if not contesto_parts:
            contesto_parts.append("Questa è una delle tue prime riflessioni - stiamo costruendo insieme la tua cronologia cognitiva in ChromaDB!")
        
        return "\n".join(contesto_parts)

    async def refract_thought(self, input_text: str) -> Dict[str, Any]:
        """Rifrazione con ChromaDB integrato e gestione errori"""
        
        self.print_header(input_text)
        
        # Test connessione
        if not await self.test_connection():
            print(f"❌ Impossibile connettersi a {self.ollama_host}")
            print("💡 Assicurati che Ollama sia avviato: ollama serve")
            return {"error": "Connection failed"}
        
        # Test modello
        model_available = await self.check_model_availability(self.model)
        if not model_available:
            print(f"⚠️ Modello '{self.model}' non disponibile")
            print(f"💡 Scarica con: ollama pull {self.model}")
            print("🔄 Continuo comunque, ma potrebbero esserci errori...")
        
        print("📚 Interrogando ChromaDB per connessioni...")
        memoria_contesto = self.prepara_contesto_memoria(input_text)
        
        pensieri_simili = self.memoria.trova_pensieri_simili(input_text, limite=2)
        if pensieri_simili:
            print(f"🔗 Trovati {len(pensieri_simili)} pensieri correlati nella tua cronologia")
        else:
            print("🆕 Nessun pensiero simile trovato - espandiamo la tua memoria!")
        
        results = {
            "original_thought": input_text,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "spectrum": {},
            "session_id": self.session_id,
            "memoria_contesto": memoria_contesto,
            "pensieri_simili_count": len(pensieri_simili),
            "errors": []
        }
        
        accumulated_context = ""
        
        for i, (lens_key, lens) in enumerate(self.cognitive_lenses.items(), 1):
            print(f"\n{lens['emoji']} {lens['name']} ({i}/6)")
            print("─" * 50)
            
            formatted_prompt = lens["prompt"].format(
                input_text=input_text,
                original_input=input_text,
                previous_output=accumulated_context,
                memoria_contesto=memoria_contesto
            )
            
            print("🤔 Riflettendo con la cronologia ChromaDB...")
            
            response = await self.call_ollama(formatted_prompt)
            
            if response.startswith("❌"):
                print(f"💡 {response}")
                results["errors"].append(f"{lens['name']}: {response}")
                response = f"[Errore nella lente {lens['name']} - verifica la configurazione Ollama]"
            else:
                print("💡", response)
            
            results["spectrum"][lens_key] = {
                "output": response,
                "order": i,
                "timestamp": datetime.now().isoformat()
            }
            
            accumulated_context += f"\n\n{lens['name']}: {response}"
            await asyncio.sleep(1.2)
        
        # Salva in ChromaDB solo se non ci sono troppi errori
        error_count = len(results["errors"])
        if self.memoria.collection and error_count < 4:  # Salva se meno di 4 errori su 6
            print(f"\n📚 Salvando in ChromaDB...")
            self.memoria.salva_pensiero(
                input_text, 
                results["spectrum"], 
                self.session_id, 
                self.model, 
                self.temperature
            )
        elif error_count >= 4:
            print(f"\n⚠️ Troppi errori ({error_count}/6) - salvataggio ChromaDB saltato")
        
        print("\n" + "=" * 70)
        if error_count == 0:
            encouragement = random.choice(self.encouragements)
            print(f"✨ {encouragement}")
            print("🌟 Rifrazione completata e memorizzata!")
        elif error_count < 3:
            print(f"⚠️ Rifrazione completata con alcuni errori ({error_count}/6)")
            print("💡 Controlla la configurazione Ollama per risultati migliori")
        else:
            print(f"❌ Rifrazione completata con molti errori ({error_count}/6)")
            print("🔧 Verifica: ollama serve && ollama pull mistral:latest")
        print("=" * 70)
        
        self.session_history.append(results)
        return results

    def show_memory_stats(self):
        """Statistiche ChromaDB"""
        stats = self.memoria.get_statistiche()
        
        print(f"\n📚 STATISTICHE CHROMADB")
        print("─" * 40)
        
        if "errore" in stats:
            print(f"❌ Errore: {stats['errore']}")
            return
        
        print(f"💾 Database: {stats.get('database_path', 'N/A')}")
        print(f"📝 Pensieri totali: {stats['pensieri_totali']}")
        
        if stats.get('primo_pensiero'):
            try:
                primo = datetime.fromisoformat(stats['primo_pensiero']).strftime("%d/%m/%Y %H:%M")
                print(f"📅 Primo pensiero: {primo}")
            except:
                print(f"📅 Primo pensiero: {stats['primo_pensiero'][:10]}")
        
        if stats.get('ultimo_pensiero'):
            try:
                ultimo = datetime.fromisoformat(stats['ultimo_pensiero']).strftime("%d/%m/%Y %H:%M")
                print(f"🕐 Ultimo pensiero: {ultimo}")
            except:
                print(f"🕐 Ultimo pensiero: {stats['ultimo_pensiero'][:10]}")

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Salva risultati JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            thought_preview = results["original_thought"][:30].replace(" ", "_")
            filename = f"prisma_chroma_{thought_preview}_{timestamp}.json"
        
        Path("./rifrazioni").mkdir(exist_ok=True)
        filepath = Path("./rifrazioni") / filename
        
        results["metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "model_used": self.model,
            "temperature": self.temperature,
            "has_chromadb": True,
            "chroma_path": str(self.memoria.chroma_path),
            "errors_count": len(results.get("errors", []))
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 File JSON salvato: {filepath}")
        return filepath

    def show_session_summary(self):
        """Riassunto sessione"""
        if not self.session_history:
            print("📝 Nessuna rifrazione in questa sessione.")
            return
            
        print(f"\n📊 RIASSUNTO SESSIONE")
        print("─" * 40)
        print(f"🔢 Rifrazioni completate: {len(self.session_history)}")
        print(f"🆔 Sessione ID: {self.session_id}")
        
        total_errors = sum(len(session.get("errors", [])) for session in self.session_history)
        if total_errors > 0:
            print(f"⚠️ Errori totali: {total_errors}")
        
        print(f"\n🎯 Pensieri analizzati:")
        for i, session in enumerate(self.session_history, 1):
            thought = session["original_thought"][:50]
            if len(session["original_thought"]) > 50:
                thought += "..."
            connessioni = session.get("pensieri_simili_count", 0)
            errori = len(session.get("errors", []))
            status = "✅" if errori == 0 else f"⚠️({errori})"
            print(f"  {i}. {thought} ({connessioni} connessioni) {status}")


async def main():
    """Main con gestione errori migliorata"""
    print("🌈 PRISMA COGNITIVO + CHROMADB (VERSIONE FIXED)")
    print("=" * 70)
    print("✨ Sistema di rifrazione cognitiva con gestione errori migliorata")
    print("🔧 Fix per ChromaDB e diagnostica Ollama avanzata")
    print("=" * 70)
    
    # Chiedi path ChromaDB
    default_path = "./chroma_journal"
    print(f"🗂️  Path ChromaDB (Enter per default: {default_path}):")
    chroma_path = input("📁 ").strip()
    if not chroma_path:
        chroma_path = default_path
    
    # Chiedi modello con suggerimenti
    print(f"\n🤖 Modello Ollama:")
    print("   Consigliati: mistral:latest, gemma:latest, llama3.1:latest")
    print("   (Enter per mistral:latest)")
    model = input("🔧 ").strip()
    if not model:
        model = "mistral:latest"
    
    # Inizializza Prisma
    prisma = PrismaCognitivoChromaFixed(
        model=model,
        chroma_path=chroma_path
    )
    
    # Diagnostica completa
    print("\n🔍 DIAGNOSTICA SISTEMA:")
    print("─" * 30)
    
    # Test Ollama
    print("🔗 Test connessione Ollama...")
    ollama_ok = await prisma.test_connection()
    if ollama_ok:
        print("✅ Ollama connesso!")
        
        # Test modello
        print(f"🔍 Test modello {model}...")
        model_ok = await prisma.check_model_availability(model)
        if model_ok:
            print(f"✅ Modello {model} disponibile!")
        else:
            print(f"❌ Modello {model} NON trovato!")
            print(f"💡 Scarica con: ollama pull {model}")
            
            # Suggerisci modelli alternativi
            print("\n🔄 Vuoi provare un modello diverso? (gemma:latest/llama3.1:latest)")
            alt_model = input("🤖 ").strip()
            if alt_model:
                alt_ok = await prisma.check_model_availability(alt_model)
                if alt_ok:
                    prisma.model = alt_model
                    print(f"✅ Cambiato a {alt_model}")
                else:
                    print(f"❌ Anche {alt_model} non disponibile")
    else:
        print("❌ Ollama NON connesso!")
        print("💡 Avvia con: ollama serve")
        print("\n🤔 Vuoi continuare comunque? (s/n)")
        if input("❓ ").lower() not in ['s', 'si', 'y', 'yes']:
            return
    
    # Test rapido del modello selezionato
    if ollama_ok:
        print(f"\n🧪 Test rapido di {prisma.model}...")
        test_response = await prisma.call_ollama("Ciao, rispondi solo con 'OK' se funzioni.")
        if "❌" not in test_response:
            print("✅ Modello risponde correttamente!")
        else:
            print(f"⚠️ Problema: {test_response}")
    
    # Mostra statistiche ChromaDB
    prisma.show_memory_stats()
    
    print(f"\n💡 COMANDI DISPONIBILI:")
    print("  📝 Scrivi un pensiero per rifrazione con memoria ChromaDB")
    print("  🔧 'model [nome]' - Cambia modello AI")
    print("  🌡️  'temp [0.0-1.0]' - Regola creatività")
    print("  🔍 'cerca [tema]' - Cerca pensieri per tema")
    print("  📚 'memory' - Statistiche ChromaDB")
    print("  🩺 'test' - Test diagnostico Ollama")
    print("  📊 'summary' - Riassunto sessione")
    print("  🚪 'quit' - Termina sessione")
    
    # Warning sui componenti mancanti
    if not HAS_CHROMA:
        print("\n⚠️  ATTENZIONE: ChromaDB non installato!")
        print("   Installa con: pip install chromadb")
    
    if not HAS_EMBEDDINGS:
        print("\n⚠️  INFO: Per ricerca semantica ottimale:")
        print("   pip install sentence-transformers")
    
    # Esempi personalizzati
    print(f"\n🎨 ESEMPI per testare:")
    print("  • Quali sono i temi ricorrenti nei miei pensieri?")
    print("  • Come posso migliorare la mia produttività?")
    print("  • Che direzione dovrei prendere nella mia carriera?")
    
    while True:
        try:
            print(f"\n💭 Condividi il tuo pensiero:")
            user_input = input("🎯 ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q', 'esci']:
                prisma.show_session_summary()
                prisma.show_memory_stats()
                print("\n👋 La tua cronologia cognitiva continua a crescere!")
                print("🌟 A presto per nuove riflessioni!")
                break
            
            if user_input.startswith('model '):
                new_model = user_input.replace('model ', '').strip()
                
                # Test del nuovo modello
                print(f"🔍 Test {new_model}...")
                model_ok = await prisma.check_model_availability(new_model)
                if model_ok:
                    prisma.model = new_model
                    print(f"✅ Modello cambiato a: {new_model}")
                    
                    # Test rapido
                    test = await prisma.call_ollama("Test rapido, rispondi 'OK'")
                    if "❌" not in test:
                        print("✅ Nuovo modello operativo!")
                    else:
                        print(f"⚠️ Problema: {test}")
                else:
                    print(f"❌ Modello {new_model} non disponibile")
                    print(f"💡 Scarica con: ollama pull {new_model}")
                continue
            
            if user_input.startswith('temp '):
                try:
                    new_temp = float(user_input.replace('temp ', '').strip())
                    if 0.0 <= new_temp <= 1.0:
                        prisma.temperature = new_temp
                        print(f"🌡️ Temperatura: {new_temp}")
                        if new_temp < 0.3:
                            print("❄️ Modalità analitica - Connessioni precise")
                        elif new_temp > 0.7:
                            print("🔥 Modalità creativa - Connessioni creative")
                        else:
                            print("⚖️ Modalità bilanciata")
                    else:
                        print("❌ Temperatura deve essere 0.0-1.0")
                except:
                    print("❌ Formato non valido. Usa: temp 0.7")
                continue
            
            if user_input.lower() == 'test':
                print("🩺 DIAGNOSTICA COMPLETA:")
                print("─" * 30)
                
                # Test Ollama
                ollama_ok = await prisma.test_connection()
                print(f"🔗 Ollama: {'✅' if ollama_ok else '❌'}")
                
                if ollama_ok:
                    # Test modello
                    model_ok = await prisma.check_model_availability(prisma.model)
                    print(f"🤖 Modello {prisma.model}: {'✅' if model_ok else '❌'}")
                    
                    # Test risposta
                    test_resp = await prisma.call_ollama("Rispondi con una sola parola: 'FUNZIONA'")
                    if "FUNZIONA" in test_resp.upper():
                        print("💬 Risposta AI: ✅")
                    else:
                        print(f"💬 Risposta AI: ❌ ({test_resp[:50]}...)")
                
                # Test ChromaDB
                if prisma.memoria.collection:
                    count = prisma.memoria.collection.count()
                    print(f"📚 ChromaDB: ✅ ({count} documenti)")
                else:
                    print("📚 ChromaDB: ❌")
                
                continue
            
            if user_input.startswith('cerca '):
                tema = user_input.replace('cerca ', '').strip()
                if prisma.memoria.collection:
                    print(f"🔍 Cerco pensieri su: '{tema}'...")
                    risultati = prisma.memoria.cerca_per_tema(tema, limite=5)
                    
                    if risultati:
                        print(f"\n📚 TROVATI {len(risultati)} PENSIERI su '{tema}':")
                        for i, r in enumerate(risultati, 1):
                            try:
                                data = datetime.fromisoformat(r['timestamp']).strftime("%d/%m/%Y")
                            except:
                                data = "Data sconosciuta"
                            
                            relevanza = int(r['relevanza'] * 100)
                            contenuto = r['contenuto'][:80] + "..." if len(r['contenuto']) > 80 else r['contenuto']
                            
                            print(f"\n{i}. [{data}] ({relevanza}% rilevante)")
                            print(f"   {contenuto}")
                            if r.get('lenti_usate'):
                                print(f"   🔍 Lenti usate: {r['lenti_usate']}")
                    else:
                        print(f"🤷 Nessun pensiero trovato su '{tema}'")
                        print("💡 Prova con parole chiave diverse!")
                else:
                    print("❌ ChromaDB non disponibile per la ricerca")
                continue
            
            if user_input.lower() == 'memory':
                prisma.show_memory_stats()
                
                if prisma.memoria.collection:
                    recenti = prisma.memoria.get_pensieri_recenti(giorni=7)
                    if recenti:
                        print(f"\n📅 PENSIERI RECENTI (ultimi 7 giorni):")
                        for i, p in enumerate(recenti[:5], 1):
                            try:
                                timestamp = datetime.fromisoformat(p['timestamp']).strftime("%d/%m %H:%M")
                            except:
                                timestamp = "Data sconosciuta"
                            
                            contenuto = p['contenuto'][:60] + "..." if len(p['contenuto']) > 60 else p['contenuto']
                            modello = p.get('modello', 'N/A')
                            print(f"  {i}. [{timestamp}] {contenuto}")
                            print(f"     🤖 {modello}")
                continue
            
            if user_input.lower() == 'summary':
                prisma.show_session_summary()
                continue
            
            # Processa il pensiero
            print(f"\n🌈 Avvio rifrazione con memoria ChromaDB...")
            results = await prisma.refract_thought(user_input)
            
            # Mostra risultati
            if "error" not in results:
                error_count = len(results.get("errors", []))
                
                if error_count == 0:
                    print("\n🎉 Rifrazione perfetta!")
                elif error_count < 3:
                    print(f"\n⚠️ Rifrazione con alcuni problemi ({error_count} errori)")
                    print("💡 Considera di controllare il modello Ollama")
                
                # Chiedi se salvare JSON
                save_choice = input("\n💾 Vuoi salvare anche un file JSON di backup? (s/n): ").lower()
                if save_choice in ['s', 'si', 'y', 'yes']:
                    prisma.save_results(results)
                
                # Suggerisci ricerche correlate
                if prisma.memoria.collection and error_count < 3:
                    print("\n🔗 Suggerimento: prova 'cerca [tema]' per esplorare pensieri correlati!")
            
            # Chiedi se continuare
            continue_choice = input("\n🔄 Vuoi esplorare un altro pensiero? (s/n): ").lower()
            if continue_choice in ['n', 'no']:
                prisma.show_session_summary()
                prisma.show_memory_stats()
                print("\n🌟 Sessione completata! ChromaDB aggiornato!")
                break
                
        except KeyboardInterrupt:
            print("\n\n🌙 Sessione interrotta.")
            prisma.show_session_summary()
            prisma.show_memory_stats()
            break
        except Exception as e:
            print(f"❌ Errore imprevisto: {e}")
            print("💡 Continuo comunque...")

if __name__ == "__main__":
    print("🚀 Avvio Prisma Cognitivo Fixed...")
    asyncio.run(main())