import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Github, ExternalLink, Eye, ChevronLeft, ChevronRight, Info, Youtube } from 'lucide-react';
import { Button } from './ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { motion, AnimatePresence } from 'framer-motion';
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";

const projectCategories = ["All", "Easy", "Medium", "Hard", "Advanced"];

const projects = [
  // All existing projects:
  /*{
    id: 1,
    title: "Next Word Predictor",
    description: "LSTM-based NLP model that predicts the most likely word following a given text sequence.",
    image: "/lovable-uploads/word_next.png",
    category: "Hard",
    type: "Core AI / ML / NLP",
    technologies: ["TensorFlow", "Keras", "Streamlit", "LSTM"],
    githubUrl: "https://github.com/wigjatin/next_word_prediction/",
    demoUrl:  "https://nextwordprediction-jatinwig.streamlit.app/",
    comment: "Built a model that finishes my sentences better than my friends do!",
    details: "This project implements a next-word prediction system using a custom-trained LSTM neural network. Trained on a custom English text corpus, it uses Keras for tokenization, sequence creation, and model building. The system processes text input, learns contextual relationships, and predicts the most probable next word out of the vocabulary. Includes top-5 ranked suggestions and a live Streamlit demo for real-time interaction. Compared LSTM with GRU and Simple RNN — LSTM outperformed both in contextual accuracy and sequence memory."
  },*/
  {
    id: 2,
    title: "Brain Tumor Detector",
    description: "Deep learning system that classifies brain MRI scans into tumor types or no tumor using transfer learning.",
    image: "/lovable-uploads/brain.png",
    category: "Hard",
    type: "Core AI / ML / NLP",
    technologies: ["TensorFlow", "Streamlit", "VGG16"],
    githubUrl: "https://github.com/wigjatin/Brain-Tumor-Detection",
    demoUrl: "https://braintumordetectionjatinwig.streamlit.app/",
    comment: "Nothing felt more powerful than teaching a model to spot tumors with 98% accuracy.",
    details: "This project uses VGG16-based transfer learning to classify brain MRI scans into glioma, meningioma, pituitary tumors, or no tumor. The model was trained on a dataset of over 2800 images, augmented for robustness, and fine-tuned using custom classification layers. It achieved 98% accuracy on test data with balanced precision and recall. A Streamlit web app provides an intuitive interface for real-time prediction. Designed to support medical professionals by offering fast, reliable tumor detection in MRI images.",
    youtubeUrl: "https://www.youtube.com/watch?v=99SklHe7cYw"
  },/*
  {
    id: 3,
    title: "Traffic Sign Recognition",
    description: "A deep learning–based system that classifies and recognizes traffic signs using TensorFlow, OpenCV, and Streamlit.",
    image: "/lovable-uploads/traffic.png",
    category: "Hard",
    type: "Core AI / ML / NLP",
    technologies: ["TensorFlow", "CNN", "OpenCV"],
    githubUrl: "https://github.com/jatinwig/traffic-signal-recognition",
    demoUrl: "https://traffic-signal-recognition-jatin-wig.streamlit.app/",
    comment: "Built for the road, but deployed in the cloud.",
    details: "This project implements a Convolutional Neural Network (CNN) to classify traffic signs with preprocessing using OpenCV. The model is deployed via Streamlit for interactive testing, allowing users to upload images and view predictions instantly."
  },*/
  {
    id: 4,
    title: "Small Language Model from scratch",
    description: "Decoder-only Transformer language model built entirely in PyTorch with an optimized training pipeline and autoregressive text generation.",
    image: "/lovable-uploads/Small-Language-Models.jpg",
    category: "Advanced",
    type: "Core AI / ML / NLP",
    technologies: ["Python", "PyTorch", "Transformers", "Mixed Precision Training", "Cosine LR Scheduling"],
    githubUrl: "https://github.com/jatin-wig/Small-Language-Model-from-scratch",
    demoUrl: "#",
    comment: "Turns out building a language model from scratch teaches you more than just tokens, it teaches patience.",
    details: "This project implements a Small Language Model completely from scratch using PyTorch, without relying on high-level training frameworks. It features a decoder-only Transformer architecture designed for efficient sequence modeling and autoregressive text generation. The training pipeline includes an optimized memmap data loader for handling large datasets, mixed-precision training for faster computation and reduced memory usage, and cosine learning rate scheduling for stable convergence. The model demonstrates a deep understanding of transformer internals, training mechanics, and scalable NLP system design.",
    youtubeUrl: "#"
  },
  {
    id: 5,
    title: "Spam Detection System",
    description: "Spam detection model using TF-IDF and Naive Bayes for accurate message classification.",
    image: "/lovable-uploads/spam.png",
    category: "Easy",
    type: "Core AI / ML / NLP",
    technologies: ["Python", "Scikit-learn", "Pandas", "Naive Bayes", "NLP"],
    githubUrl: "https://github.com/wigjatin/spam-detector-large-scale",
    demoUrl: "https://spam-detection-jatin-wig.streamlit.app/",
    comment: "Never thought I'd get excited over a message being labeled 'spam' — here we are.",
    details: "This spam detection system uses Natural Language Processing (NLP) and machine learning to accurately classify messages as spam or not spam. It leverages TF-IDF vectorization and the Multinomial Naive Bayes algorithm, focusing on high precision to avoid false positives. The model is trained on real-world datasets and includes a prediction script for real-time message filtering.",
    youtubeUrl: "https://www.youtube.com/watch?v=CEqrNU5ZLVI"
  },
  {
    id: 6,
    title: "TradeValidator AI",
    description: "Vision-powered trading intelligence system that analyzes chart screenshots and generates structured trade insights.",
    image: "/lovable-uploads/TradeValidator.png",
    category: "Hard",
    type: "GenAI",
    technologies: ["Python", "Google Gemini API", "Computer Vision", "Prompt Engineering",],
    githubUrl: "https://github.com/jatin-wig/Trade-Validator-AI",
    demoUrl: "#",
    comment: "Because sometimes a second pair of AI eyes can spot what human emotions miss.",
    details: "TradeValidator AI is a vision-powered trading intelligence platform that analyzes chart screenshots to produce professional, structured trade insights. Leveraging multimodal AI, the system evaluates market structure, trade validity, entry quality, and potential risk conditions to support more informed decision-making. The platform focuses on objective analysis rather than prediction, helping traders reduce emotional bias while reviewing setups. Built with scalability in mind, it demonstrates practical applications of computer vision and large language models within the fintech space. This tool is strictly intended for educational and informational purposes only and does not provide financial advice.",
    youtubeUrl: "https://www.youtube.com/watch?v=HjinGwvfGeQ"
  },
  {
    id: 7,
    title: "Food Delivery Customer Support",
    description: "AI-powered customer support platform that combines deterministic workflows with LLM intelligence to simulate real-world food delivery operations.",
    image: "/lovable-uploads/food.png",
    category: "Medium",
    type: "GenAI",
    technologies: ["Python", "FastAPI", "Streamlit", "SQLite", "LLMs", "Workflow Automation"],
    githubUrl: "https://github.com/jatin-wig/Food-Delivery-Customer-Support",
    demoUrl: "#",
    comment: "Turns customer chaos into structured conversations — automatically.",
    details: "The AI Food Delivery Support System is an intelligent platform designed to replicate real-world customer support operations within a food delivery ecosystem. It combines deterministic backend workflows with large language model intelligence to manage the complete order lifecycle — including ETA estimation, cancellations, refunds, and dynamic customer interactions. Built using FastAPI for high-performance APIs, Streamlit for an interactive interface, and SQLite for lightweight data persistence, the system demonstrates production-style architecture and orchestration. By blending rule-based logic with conversational AI, the project showcases how modern support platforms can scale efficiently while maintaining contextual, human-like communication.",
    youtubeUrl: "https://www.youtube.com/watch?v=BA6S7_kaFD0"
  },
  {
    id: 8,
    title: "House Price Predictor",
    description: "Machine learning model that estimates real estate prices based on key property features.",
    image: "/lovable-uploads/house.jpg",
    category: "Medium",
    type: "Core AI / ML / NLP",
    technologies: ["Python", "Scikit-learn", "Gradient Boosting", "Pandas"],
    githubUrl: "https://github.com/wigjatin/House-prediction-model/tree/main",
    demoUrl: "https://house-prediction-model-jatin-wig.streamlit.app/",
    comment: "Taught a model to value homes — it doesn't even have one.",
    details: "This project applies Gradient Boosting Regression to predict property values using historical sales data and structural attributes like square footage, quality rating, and year built. The pipeline includes robust preprocessing with outlier filtering, feature scaling, and log-transformed targets for improved accuracy. It models complex, non-linear feature interactions and outputs highly reliable estimates, making it useful for buyers, sellers, and agents alike.",
    youtubeUrl: "https://www.youtube.com/watch?v=SEAnfUnSGbU"
  },
  {
    id: 9,
    title: "Document Intelligence — PDF Chat",
    description: "Production-style local RAG system enabling conversational querying and semantic search over PDF documents using Ollama.",
    image: "/lovable-uploads/document.png",
    category: "Medium",
    type: "GenAI",
    technologies: ["Python", "Ollama", "Llama3", "LangChain", "FAISS", "HuggingFace", "Streamlit"],
    githubUrl: "https://github.com/jatin-wig/Document-Intelligence-PDF-Chat-with-Ollama",
    demoUrl: "#",
    comment: "Your documents shouldn’t just sit there, they should be able to talk back.",
    details: "Document Intelligence — PDF Chat is a production-style Retrieval-Augmented Generation (RAG) system designed for fully local inference, enabling secure and private conversational interactions with PDF documents. The application ingests structured documents, generates high-quality vector embeddings using HuggingFace Sentence Transformers, and performs semantic retrieval through FAISS. With Maximum Marginal Relevance (MMR) optimization, the system enhances context diversity while minimizing redundancy in retrieved chunks. Powered by a locally hosted Llama3 model via Ollama, it eliminates reliance on external LLM APIs while maintaining strong reasoning capabilities. Built with LangChain for orchestration and Streamlit for an interactive UI, the project reflects real-world GenAI architecture focused on privacy, performance, and scalable document intelligence workflows.",
    youtubeUrl: "https://www.youtube.com/watch?v=taKxQnO6HM0"
  },
  {
    id: 10,
    title: "YouTube Transcript Chatbot",
    description: "GenAI chatbot that enables conversational querying over YouTube videos by transforming transcripts into a searchable knowledge base.",
    image: "/lovable-uploads/youtube.png",
    category: "Hard",
    type: "GenAI",
    technologies: ["Python", "Streamlit", "LangChain", "FAISS", "Sentence Transformers", "Google Gemini", "youtube-transcript-api"],
    githubUrl: "https://github.com/jatin-wig/Youtube-Transcription-Gen-AI",
    demoUrl: "#",
    comment: "Because watching a 2-hour video for one answer is inefficient , querying it is smarter.",
    details: "The YouTube Transcript Chatbot is a Streamlit-based generative AI application that allows users to chat with any YouTube video using its transcript as the knowledge source. The system automatically extracts transcripts with timestamps, converts them into vector embeddings, and stores them in FAISS for high-speed semantic retrieval. Built on a Retrieval-Augmented Generation (RAG) pipeline with LangChain, it generates grounded responses through Google Gemini while maintaining full conversational memory. The application enhances transparency by displaying sources for each answer and dynamically generating clickable timestamp links that redirect users to the exact moment in the video. This project demonstrates strong capabilities in RAG architecture, vector search optimization, conversational context handling, and production-style GenAI application design.",
    youtubeUrl: "https://www.youtube.com/watch?v=N2HJ-i1qreA"
  },
  {
    id: 11,
    title: "Resume + Job Assistant",
    description: "Gemini-powered resume intelligence platform that analyzes resumes against job descriptions and generates ATS-driven improvements and tailored cover letters.",
    image: "/lovable-uploads/resume.png",
    category: "Medium",
    type: "GenAI",
    technologies: ["Python", "Streamlit", "Google Gemini", "PDF Parsing", "Prompt Engineering"],
    githubUrl: "https://github.com/jatin-wig/Resume-Job-Assistant-Gen-AI-",
    demoUrl: "#",
    comment: "Your resume deserves better odds.",
    details: "Resume + Job Assistant is a generative AI platform designed to streamline the job application process through intelligent resume analysis. Users can upload a resume PDF and compare it against a job description to receive ATS-focused optimization tips, keyword gap analysis, alignment insights, and actionable resume improvements. The system also generates job-specific cover letters by synthesizing resume data with role requirements. Powered by Google Gemini and delivered through a clean Streamlit interface with dedicated feature tabs, the application demonstrates practical LLM orchestration, structured prompt design, and real-world NLP workflows aimed at increasing application effectiveness.",
    youtubeUrl: "https://youtu.be/I67IT8pcW9A"
  },
  {
    id: 12,
    title: "Twitter Sentiment Analysis",
    description: "Lightweight NLP-powered system that classifies tweets into positive or negative sentiments using Logistic Regression",
    image: "/lovable-uploads/twitter.png",
    category: "Hard",
    type: "Core AI / ML / NLP",
    technologies: ["Python", "NLP", "Logistic Regression", "TF-IDF", "Scikit-learn", "Streamlit"],
    githubUrl: "https://github.com/jatin-wig/Twitter-Sentiment-Analysis",
    demoUrl: "https://twitter-sentiment-analysis-by-jatinwig.streamlit.app/",
    comment: "Transforming chaotic tweets into clear sentiment insights!",
    details: "This project is a lightweight NLP-powered sentiment analysis system designed to classify tweets as either positive or negative. It uses TF-IDF vectorization to convert text into meaningful numerical features and applies a Logistic Regression model for fast, interpretable predictions. The preprocessing pipeline includes tokenization, normalization, and stopword removal to improve model accuracy. Deployed as an interactive Streamlit web app, the system enables real-time tweet analysis, making it valuable for businesses tracking customer feedback, researchers studying public opinion, and individuals seeking structured insights from social media data.",
    youtubeUrl: "https://www.youtube.com/watch?v=GTBRKzo82yw"
  },
  {
    id: 13,
    title: "Placement Prediction System",
    description: "Web app that predicts student placement outcomes based on academic and background data.",
    image: "/lovable-uploads/placement.png",
    category: "Easy",
    type: "Core AI / ML / NLP",
    technologies: ["Python", "Logistic Regression", "Scikit-learn", "Pandas"],
    githubUrl: "https://github.com/wigjatin/Student---Placement--Prediction-/tree/main",
    demoUrl: "https://student-placement-prediction-jatin-wig.streamlit.app/",
    comment: "Interview anxiety meets predictive analytics.",
    details: "This application uses a Logistic Regression model trained on historical student data to predict whether a student is likely to be placed. It takes inputs such as academic performance, work experience, and degree specialization to estimate placement probability. The app offers a clean and interactive Streamlit interface, making it useful for students, counselors, and career services to assess employability based on quantitative factors. The model was built with scikit-learn and deployed using joblib for efficient performance.",
    youtubeUrl: "https://www.youtube.com/watch?v=9r151tjA4eg"
  },

  /*{
    id: 15,
    title: "Music Genre Classifier",
    description: "Audio analysis tool that identifies music genres from short audio samples.",
    image: "https://images.unsplash.com/photo-1485827404703-89b55fcc595e",
    category: "Easy",
    type: "Core AI / ML / NLP",
    technologies: ["Librosa", "CNN", "Flask"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "Still argues with me about whether that one song is indie rock or alternative.",
    details: "This audio analysis tool can identify music genres from just a few seconds of audio. It uses convolutional neural networks to analyze spectrograms and other audio features extracted with Librosa. The classifier can distinguish between 10 popular music genres with 85% accuracy and provides confidence scores for each potential classification."
  },
  {
    id: 16,
    title: "Reinforcement Learning Game AI",
    description: "Self-learning AI agent that masters complex games through reinforcement learning.",
    image: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5",
    category: "Hard",
    type: "Core AI / ML / NLP",
    technologies: ["PyTorch", "OpenAI Gym", "DQN"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "It beat me at chess after just 4 hours of self-play training!",
    details: "This reinforcement learning agent can master various games by playing against itself or other opponents. Using Deep Q-Networks (DQN) and policy gradient methods, it learns optimal strategies through trial and error. The system includes visualization tools to understand the agent's learning process and decision-making strategies. It has successfully learned to play chess, Go, and several Atari games."
  },
  {
    id: 17,
    title: "Personal Finance Advisor",
    description: "AI-powered financial advisor that provides personalized savings and investment recommendations.",
    image: "https://images.unsplash.com/photo-1461749280684-dccba630e2f6",
    category: "Medium",
    type: "Core AI / ML / NLP",
    technologies: ["Scikit-learn", "NLP", "Flask"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "Convinced me to stop buying expensive coffee. I'm richer but grumpier now.",
    details: "This personal finance advisor analyzes spending patterns, income, and financial goals to provide tailored recommendations for savings and investments. It categorizes transactions automatically, identifies spending trends, and suggests actionable changes to help users meet their financial objectives. The system includes scenario modeling to show the long-term impact of different financial decisions."
  },
  {
    id: 18,
    title: "Markdown Note Taker",
    description: "Simple note-taking app with AI-powered organization and tagging.",
    image: "https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d",
    category: "Easy",
    type: "Core AI / ML / NLP",
    technologies: ["React", "NLP", "Local Storage"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "Finally, an app that understands my chaotic note-taking style!",
    details: "This markdown note-taking app automatically organizes and tags your notes based on content analysis. Using natural language processing, it identifies key topics, related concepts, and action items within your notes. The system suggests connections between different notes and helps users build a personal knowledge graph. All processing happens locally for privacy."
  },
  {
    id: 19,
    title: "Satellite Image Analyzer",
    description: "Deep learning tool that analyzes satellite imagery to detect environmental changes.",
    image: "https://images.unsplash.com/photo-1518770660439-4636190af475",
    category: "Hard",
    type: "Core AI / ML / NLP",
    technologies: ["TensorFlow", "U-Net", "GIS"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "Spotted deforestation activity that had been missed by manual monitoring.",
    details: "This deep learning tool analyzes satellite imagery to detect and quantify environmental changes such as deforestation, urban expansion, and crop health. Using U-Net architecture for semantic segmentation, it can process large geographical areas efficiently and highlight regions of interest. The system integrates with GIS platforms and provides time-series analysis to track changes over months or years."
  },
  {
    id: 20,
    title: "AI Art Generator",
    description: "Creative AI that generates original artwork based on text descriptions or style references.",
    image: "https://images.unsplash.com/photo-1531297484001-80022131f5a1",
    category: "Advanced",
    type: "GenAI",
    technologies: ["GANs", "CLIP", "PyTorch"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "Its Van Gogh-inspired space scenes are now hanging in my living room!",
    details: "This creative AI generates original artwork based on text descriptions or style references. Combining Generative Adversarial Networks (GANs) with CLIP for text-to-image understanding, it can interpret complex prompts like 'sunset over mountains in the style of impressionism'. The system allows fine control over various aspects of the generated images and can produce high-resolution outputs suitable for printing."
  },
  {
    id: 21,
    title: "Social Media Trend Predictor",
    description: "AI tool that predicts emerging social media trends before they go viral.",
    image: "https://images.unsplash.com/photo-1487058792275-0ad4aaf24ca7",
    category: "Medium",
    type: "Core AI / ML / NLP",
    technologies: ["NLP", "Time Series", "Graph Neural Networks"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "Predicted three major memes a week before they blew up!",
    details: "This predictive tool analyzes patterns in social media content and engagement to identify emerging trends before they go viral. Using a combination of natural language processing and graph neural networks to model content spread patterns, it can detect subtle signals that indicate potential viral content. The system provides early alerts for brand marketers and content creators to capitalize on rising trends."
  },
  {
    id: 22,
    title: "DIY Smart Mirror",
    description: "Raspberry Pi-powered smart mirror with face recognition and personalized information display.",
    image: "https://images.unsplash.com/photo-1649972904349-6e44c42644a7",
    category: "Easy",
    type: "Core AI / ML / NLP",
    technologies: ["Python", "OpenCV", "Raspberry Pi"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "Now I get weather updates while checking my hair!",
    details: "This DIY smart mirror uses a Raspberry Pi, two-way mirror, and display panel to create an interactive mirror that shows personalized information while functioning as a normal mirror. Using OpenCV for face recognition, it identifies different users and displays relevant information for each person, such as their calendar, weather, news, and traffic. The system includes voice control and customizable widgets."
  },
  {
    id: 23,
    title: "Quantum Algorithm Simulator",
    description: "Educational tool that simulates quantum computing algorithms for learning purposes.",
    image: "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158",
    category: "Hard",
    type: "Core AI / ML / NLP",
    technologies: ["Qiskit", "NumPy", "Matplotlib"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "Makes quantum computing slightly less mind-boggling...slightly.",
    details: "This educational tool simulates quantum computing algorithms to help students and researchers understand quantum principles without access to actual quantum hardware. It visualizes quantum states, circuit operations, and measurement probabilities in an interactive interface. The simulator supports common quantum algorithms like Grover's search and Shor's factoring algorithm, with step-by-step explanations of the underlying quantum mechanics."
  },
  {
    id: 24,
    title: "3D Protein Folding Visualizer",
    description: "Interactive tool that visualizes protein folding simulations in 3D space.",
    image: "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b",
    category: "Advanced",
    type: "Core AI / ML / NLP",
    technologies: ["PyMOL", "AlphaFold", "Three.js"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "Biologists said it was like seeing their research come to life!",
    details: "This interactive visualization tool renders complex protein structures in 3D space and animates the protein folding process. Integrating with AlphaFold predictions, it helps researchers understand how protein structures form and interact. The web-based interface allows users to rotate, zoom, highlight specific amino acids, and view different representations of the protein structure. The system also identifies potential binding sites and structural motifs."
  },
  {
    id: 25,
    title: "Language Learning Flashcards",
    description: "AI-enhanced flashcard app that adjusts to your learning pace and trouble areas.",
    image: "https://images.unsplash.com/photo-1498050108023-c5249f4df085",
    category: "Easy",
    type: "Core AI / ML / NLP",
    technologies: ["React", "Spaced Repetition", "NLP"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "Finally remembered the difference between 'por' and 'para' in Spanish!",
    details: "This language learning app uses spaced repetition algorithms enhanced by AI to optimize vocabulary acquisition. It tracks your performance on different words and grammatical concepts, automatically focusing more on items you struggle with. The system uses natural language processing to generate contextually relevant example sentences and provides pronunciation feedback through speech recognition."
  },
  {
    id: 26,
    title: "Neural Architecture Search",
    description: "Automated system that designs optimal neural network architectures for specific tasks.",
    image: "https://images.unsplash.com/photo-1485827404703-89b55fcc595e",
    category: "Advanced",
    type: "Core AI / ML / NLP",
    technologies: ["AutoML", "PyTorch", "Reinforcement Learning"],
    githubUrl: "#",
    demoUrl: "#",
    comment: "It designed a neural network that outperformed my hand-crafted model after just 24 hours of search!",
    details: "This automated system uses reinforcement learning and evolutionary algorithms to discover optimal neural network architectures for specific tasks. It explores the space of possible architectures efficiently, testing promising candidates on the target problem. The system optimizes for both performance and computational efficiency, creating networks that achieve state-of-the-art results while minimizing parameters and inference time."
  }*/
];

const Projects = () => {
  const [activeType, setActiveType] = useState("GenAI");
  const [activeCategory, setActiveCategory] = useState("All");

  const filteredProjects = projects
    .filter(project => project.type === activeType)
    .filter(project => activeCategory === "All" ? true : project.category === activeCategory);
  const [viewingProject, setViewingProject] = useState(null);
  const [carouselApi, setCarouselApi] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [isPaused, setIsPaused] = useState(false);

  useEffect(() => {
    if (!carouselApi) return;

    setTotalPages(carouselApi.scrollSnapList().length);
    setCurrentPage(carouselApi.selectedScrollSnap());

    const onSelect = () => {
      setCurrentPage(carouselApi.selectedScrollSnap());
    };

    const onReInit = () => {
      setTotalPages(carouselApi.scrollSnapList().length);
      setCurrentPage(carouselApi.selectedScrollSnap());
    };

    carouselApi.on("select", onSelect);
    carouselApi.on("reInit", onReInit);

    // Custom Autoplay Logic
    const autoplayInterval = setInterval(() => {
      if (!isPaused && carouselApi) {
        carouselApi.scrollNext();
      }
    }, 4000); // 4 seconds interval

    return () => {
      carouselApi.off("select", onSelect);
      carouselApi.off("reInit", onReInit);
      clearInterval(autoplayInterval);
    };
  }, [carouselApi, isPaused]);

  useEffect(() => {
    if (carouselApi) {
      carouselApi.scrollTo(0);
    }
  }, [activeType, activeCategory, carouselApi]);

  const handleCategoryChange = (value) => {
    if (value) setActiveCategory(value);
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring",
        damping: 12,
        stiffness: 100
      }
    }
  };

  const handleNext = () => {
    if (carouselApi) {
      carouselApi.scrollNext();
    }
  };

  const handlePrevious = () => {
    if (carouselApi) {
      carouselApi.scrollPrev();
    }
  };

  return (
    <section className="py-16 px-6 bg-gradient-to-b from-white to-gray-50">
      <div className="max-w-6xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, type: "spring", damping: 15 }}
          className="text-4xl md:text-5xl font-bold text-center mb-4 text-gradient font-heading tracking-tight"
        >
          My AI Projects
        </motion.h2>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.2, type: "spring", damping: 20 }}
          className="text-gray-600 text-center mb-8 text-lg"
        >
          From simple scripts to complex neural networks, here's what I've built.
        </motion.p>

        {/* Project Type Tabs */}
        <motion.div
          className="flex justify-center mb-8"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3, type: "spring", damping: 20 }}
        >
          <ToggleGroup
            type="single"
            value={activeType}
            onValueChange={(value) => value && setActiveType(value)}
            className="flex gap-2 p-1.5 bg-white/50 backdrop-blur-sm rounded-full shadow-sm border border-gray-100 dark:border-gray-800 dark:bg-gray-900/50"
          >
            <ToggleGroupItem
              value="GenAI"
              aria-label="GenAI Projects"
              className={`px-6 md:px-8 py-3 rounded-full transition-all duration-300 font-bold ${activeType === "GenAI"
                ? "bg-purple-200 text-black shadow-lg shadow-purple-500/20 ring-2 ring-purple-400/30"
                : "bg-transparent text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
                }`}
            >
              <motion.span whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>GenAI</motion.span>
            </ToggleGroupItem>
            <ToggleGroupItem
              value="Core AI / ML / NLP"
              aria-label="Core AI / ML / NLP Projects"
              className={`px-6 md:px-8 py-3 rounded-full transition-all duration-300 font-bold ${activeType === "Core AI / ML / NLP"
                ? "bg-purple-200 text-black shadow-lg shadow-purple-500/20 ring-2 ring-purple-400/30"
                : "bg-transparent text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
                }`}
            >
              <motion.span whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>Core AI / ML / NLP</motion.span>
            </ToggleGroupItem>
          </ToggleGroup>
        </motion.div>

        {/* Category Filter */}
        <motion.div
          className="flex justify-center mb-12"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4, type: "spring", damping: 20 }}
        >
          <ToggleGroup
            type="single"
            value={activeCategory}
            onValueChange={handleCategoryChange}
            className="flex flex-wrap justify-center gap-2 p-1.5 bg-gray-50 rounded-full shadow-sm"
          >
            {projectCategories.map((category) => (
              <ToggleGroupItem
                key={category}
                value={category}
                aria-label={`Filter by ${category}`}
                className={`px-6 py-2 rounded-full transition-all duration-300 font-bold ${activeCategory === category
                  ? "bg-purple-200 text-black shadow-md ring-1 ring-purple-400/30"
                  : "bg-transparent text-gray-700 hover:bg-gray-100 dark:text-gray-300"
                  }`}
              >
                <motion.span
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {category}
                </motion.span>
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, type: "spring", damping: 20 }}
          className="mb-12 relative"
          onMouseEnter={() => setIsPaused(true)}
          onMouseLeave={() => setIsPaused(false)}
        >
          {/* Constrain filtered projects results */}
          {filteredProjects.length === 0 && (
            <div className="text-center py-20 text-gray-500 italic">
              No projects found in this category.
            </div>
          )}
          <Carousel
            opts={{
              align: "start",
              loop: true,
              duration: 25,
              skipSnaps: false,
              dragFree: false,
            }}
            className="w-full"
            setApi={setCarouselApi}
          >
            <CarouselContent className="-ml-2 py-4 transition-transform duration-500 ease-out">
              {filteredProjects.map((project, index) => (
                <CarouselItem key={project.id} className="basis-full md:basis-1/2 lg:basis-1/3 pl-4">
                  <div className="h-full" data-project-id={project.id}>
                    <Card
                      className="h-full glass-panel border-none overflow-hidden group hover:-translate-y-2 transition-transform duration-500 cursor-pointer"
                      onClick={() => setViewingProject(project)}
                    >
                      <div className="relative overflow-hidden h-48">
                        <motion.div
                          whileHover={{ scale: 1.1 }}
                          transition={{ duration: 0.6 }}
                          className="w-full h-full"
                        >
                          <img
                            src={project.image}
                            alt={project.title}
                            className="w-full h-full object-cover"
                          />
                        </motion.div>
                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                        <motion.div
                          className="absolute top-4 right-4 bg-white/90 dark:bg-black/80 backdrop-blur-md px-3 py-1 rounded-full text-xs font-bold text-brand-purple shadow-sm border border-white/20"
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.2, duration: 0.3 }}
                        >
                          {project.category}
                        </motion.div>
                      </div>

                      <CardHeader className="pb-2">
                        <CardTitle className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400 group-hover:from-brand-purple group-hover:to-brand-pink transition-all duration-300">
                          {project.title}
                        </CardTitle>
                        <CardDescription className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                          {project.description}
                        </CardDescription>
                      </CardHeader>

                      <CardContent className="pb-4 flex-grow">
                        <div className="flex gap-2 mb-4 flex-wrap">
                          {project.technologies.slice(0, 3).map((tech) => (
                            <span
                              key={tech}
                              className="px-2.5 py-1 bg-purple-200 border border-purple-300 text-black text-xs font-bold rounded-full"
                            >
                              {tech}
                            </span>
                          ))}
                          {project.technologies.length > 3 && (
                            <span className="px-2.5 py-1 bg-gray-100 dark:bg-gray-800 text-gray-500 text-xs font-medium rounded-full">
                              +{project.technologies.length - 3}
                            </span>
                          )}
                        </div>

                        <p className="text-gray-500 dark:text-gray-500 italic text-xs border-l-2 border-brand-purple/30 pl-3">
                          "{project.comment}"
                        </p>
                      </CardContent>

                      <CardFooter className="pt-0 flex justify-between items-center mt-auto" onClick={(e) => e.stopPropagation()}>
                        <div className="flex space-x-2">
                          <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={(e) => {
                              e.stopPropagation();
                              setViewingProject(project);
                            }}
                            className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-purple-200 text-black font-bold text-sm hover:bg-brand-purple hover:text-white transition-colors"
                          >
                            <Eye size={16} /> View
                          </motion.button>
                        </div>
                        <div className="flex items-center space-x-2">
                          <motion.a
                            href={project.githubUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            whileHover={{ scale: 1.1, rotate: 10 }}
                            className="w-9 h-9 flex items-center justify-center rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-brand-purple hover:text-white transition-colors"
                          >
                            <Github size={18} />
                          </motion.a>
                          {project.demoUrl && project.demoUrl !== "#" && (
                            <motion.a
                              href={project.demoUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              whileHover={{ scale: 1.1, rotate: -10 }}
                              className="w-9 h-9 flex items-center justify-center rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-brand-pink hover:text-white transition-colors"
                            >
                              <ExternalLink size={18} />
                            </motion.a>
                          )}
                          {project.youtubeUrl && (
                            <motion.a
                              href={project.youtubeUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              whileHover={{ scale: 1.1, rotate: 10 }}
                              className="w-9 h-9 flex items-center justify-center rounded-full bg-gray-100 dark:bg-gray-800 text-red-600 hover:bg-red-600 hover:text-white transition-colors"
                            >
                              <Youtube size={18} />
                            </motion.a>
                          )}
                        </div>
                      </CardFooter>
                    </Card>
                  </div>
                </CarouselItem>
              ))}
            </CarouselContent>
          </Carousel>

          <div className="flex items-center justify-center mt-8 gap-8">
            <motion.button
              className="flex items-center justify-center w-12 h-12 rounded-full bg-white shadow-md hover:shadow-lg transition-all duration-300 text-brand-purple"
              onClick={handlePrevious}
              whileHover={{ scale: 1.1, backgroundColor: "#f9f5ff" }}
              whileTap={{ scale: 0.9 }}
              disabled={currentPage === 0 && !carouselApi?.options?.loop}
              style={{ opacity: currentPage === 0 && !carouselApi?.options?.loop ? 0.5 : 1 }}
            >
              <ChevronLeft className="h-6 w-6" />
            </motion.button>

            <div className="flex gap-2">
              {Array.from({ length: totalPages }).map((_, index) => (
                <motion.button
                  key={index}
                  className={`h-3 rounded-full transition-all duration-500 ease-out ${index === currentPage ? "bg-brand-purple w-10" : "bg-gray-300 w-3"
                    }`}
                  onClick={() => {
                    carouselApi?.scrollTo(index);
                  }}
                  whileHover={{ scale: 1.2 }}
                  whileTap={{ scale: 0.9 }}
                />
              ))}
            </div>

            <motion.button
              className="flex items-center justify-center w-12 h-12 rounded-full bg-white shadow-md hover:shadow-lg transition-all duration-300 text-brand-purple"
              onClick={handleNext}
              whileHover={{ scale: 1.1, backgroundColor: "#f9f5ff" }}
              whileTap={{ scale: 0.9 }}
              disabled={currentPage === totalPages - 1 && !carouselApi?.options?.loop}
              style={{ opacity: currentPage === totalPages - 1 && !carouselApi?.options?.loop ? 0.5 : 1 }}
            >
              <ChevronRight className="h-6 w-6" />
            </motion.button>
          </div>
        </motion.div>

        <AnimatePresence>
          {viewingProject && (
            <motion.div
              className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setViewingProject(null)}
            >
              <motion.div
                className="bg-white rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto shadow-2xl"
                initial={{ scale: 0.9, opacity: 0, y: 20 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.9, opacity: 0, y: 20 }}
                transition={{
                  type: "spring",
                  damping: 25,
                  stiffness: 300
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="relative">
                  <img src={viewingProject.image} alt={viewingProject.title} className="w-full h-64 object-cover rounded-t-2xl" />
                  <motion.button
                    className="absolute top-4 right-4 bg-white/80 backdrop-blur-sm w-10 h-10 rounded-full hover:bg-white flex items-center justify-center"
                    onClick={() => setViewingProject(null)}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18" /><path d="m6 6 12 12" /></svg>
                  </motion.button>
                </div>
                <div className="p-8">
                  <h3 className="text-3xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-gray-800 to-brand-purple">{viewingProject.title}</h3>
                  <div className="flex gap-2 mb-5 flex-wrap">
                    {viewingProject.technologies.map((tech) => (
                      <motion.span
                        key={tech}
                        className="px-3 py-1.5 bg-gradient-to-r from-brand-light-purple to-brand-purple/20 text-brand-purple text-sm rounded-full"
                        whileHover={{ scale: 1.1, boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)" }}
                        whileTap={{ scale: 0.95 }}
                      >
                        {tech}
                      </motion.span>
                    ))}
                  </div>
                  <p className="text-gray-700 mb-6 text-lg">{viewingProject.description}</p>

                  <div className="mb-6 bg-purple-50 dark:bg-purple-900/10 p-4 rounded-lg border border-purple-100 dark:border-purple-800/20">
                    <h4 className="flex items-center gap-2 text-sm font-bold text-brand-purple mb-2">
                      Developer Note
                    </h4>
                    <p className="text-gray-700 dark:text-gray-300 italic">"{viewingProject.comment}"</p>
                  </div>

                  <h4 className="text-xl font-semibold mb-3">Project Details</h4>
                  <p className="text-gray-600 mb-6">{viewingProject.details}</p>
                  <div className="flex justify-between items-center pt-4 border-t">
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                      <a href={viewingProject.githubUrl} className="flex items-center gap-1 text-brand-purple font-medium hover:underline">
                        <Github size={18} /> GitHub Repository
                      </a>
                    </motion.div>
                    {viewingProject.demoUrl && viewingProject.demoUrl !== "#" && (
                      <motion.div
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <Button
                          className="rounded-full bg-gradient-to-r from-brand-purple to-brand-vivid-purple hover:opacity-90 hover:shadow-lg transition-all duration-300"
                        >
                          <a href={viewingProject.demoUrl} className="flex items-center gap-1 text-white font-medium">
                            <ExternalLink size={18} /> Live Demo
                          </a>
                        </Button>
                      </motion.div>
                    )}
                    {viewingProject.youtubeUrl && (
                      <motion.div
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <Button
                          className="rounded-full bg-red-600 hover:bg-red-700 text-white hover:shadow-lg transition-all duration-300"
                        >
                          <a href={viewingProject.youtubeUrl} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1">
                            <Youtube size={18} /> Video
                          </a>
                        </Button>
                      </motion.div>
                    )}
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </section>
  );
};

export default Projects;