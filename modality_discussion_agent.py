import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModalityAgent:
    """Represents a modality agent with its characteristics and reasoning capabilities."""
    name: str
    modality: str
    primary_emotion: str
    primary_confidence: int
    data_quality_score: int
    data_quality_issues: List[str]
    reasoning: str
    emotions: List[Dict[str, Any]]
    
    def introduce_self(self) -> str:
        """Agent introduces itself with its findings."""
        issues_text = ", ".join(self.data_quality_issues) if self.data_quality_issues else "none"
        return f"I am the {self.name} agent. I detected '{self.primary_emotion}' with {self.primary_confidence}% confidence. My data quality score is {self.data_quality_score}/100 with issues: {issues_text}. My reasoning: {self.reasoning}"

    def discuss_confidence(self, other_agents: List['ModalityAgent']) -> str:
        """Agent discusses its confidence level and compares with others."""
        other_emotions = [f"{agent.name}: {agent.primary_emotion} ({agent.primary_confidence}%)" for agent in other_agents]
        other_emotions_text = "; ".join(other_emotions)
        
        if self.primary_confidence >= 80:
            confidence_level = "very high"
        elif self.primary_confidence >= 60:
            confidence_level = "high"
        elif self.primary_confidence >= 40:
            confidence_level = "moderate"
        else:
            confidence_level = "low"
            
        return f"As {self.name}, I have {confidence_level} confidence in my prediction of '{self.primary_emotion}'. Other agents found: {other_emotions_text}. My data quality of {self.data_quality_score}/100 supports my confidence level."

    def discuss_data_quality(self) -> str:
        """Agent discusses its data quality assessment."""
        if self.data_quality_score >= 80:
            quality_level = "excellent"
        elif self.data_quality_score >= 60:
            quality_level = "good"
        elif self.data_quality_score >= 40:
            quality_level = "fair"
        else:
            quality_level = "poor"
            
        issues_impact = "These issues may affect my reliability." if self.data_quality_issues else "No significant issues detected."
        
        return f"My data quality is {quality_level} ({self.data_quality_score}/100). {issues_impact}"

    def respond_to_disagreement(self, conflicting_agent: 'ModalityAgent') -> str:
        """Agent responds to disagreements with other agents."""
        if self.primary_emotion == conflicting_agent.primary_emotion:
            return f"I agree with {conflicting_agent.name} on '{self.primary_emotion}', which strengthens our collective confidence."
        
        confidence_diff = self.primary_confidence - conflicting_agent.primary_confidence
        quality_diff = self.data_quality_score - conflicting_agent.data_quality_score
        
        if confidence_diff > 20 and quality_diff > 10:
            return f"I disagree with {conflicting_agent.name}. My higher confidence ({self.primary_confidence}% vs {conflicting_agent.primary_confidence}%) and better data quality ({self.data_quality_score} vs {conflicting_agent.data_quality_score}) suggest '{self.primary_emotion}' is more likely."
        elif confidence_diff < -20 and quality_diff < -10:
            return f"I acknowledge {conflicting_agent.name}'s stronger position. Their higher confidence and data quality suggest '{conflicting_agent.primary_emotion}' may be correct, though I still see evidence for '{self.primary_emotion}'."
        else:
            return f"There's a disagreement with {conflicting_agent.name}. We have similar confidence levels, so we should consider the context and data quality factors carefully."

    def propose_consensus(self, all_agents: List['ModalityAgent']) -> str:
        """Agent proposes a consensus based on all agents' inputs."""
        emotion_counts = {}
        weighted_emotions = {}
        
        for agent in all_agents:
            emotion = agent.primary_emotion
            weight = (agent.primary_confidence / 100) * (agent.data_quality_score / 100)
            
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            weighted_emotions[emotion] = weighted_emotions.get(emotion, 0) + weight
        
        most_common = max(emotion_counts.items(), key=lambda x: x[1])
        highest_weighted = max(weighted_emotions.items(), key=lambda x: x[1])
        
        if most_common[0] == highest_weighted[0]:
            return f"Based on our discussion, I propose consensus on '{most_common[0]}' as it's both most common ({most_common[1]} agents) and highest weighted ({highest_weighted[1]:.2f})."
        else:
            return f"I see conflicting signals: '{most_common[0]}' is most common but '{highest_weighted[0]}' has highest weighted score. We need further discussion."

class ModalityDiscussionSystem:
    """Manages the discussion between modality agents."""
    
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.data = self.load_data()
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the JSON data from file."""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_agents_from_sample(self, sample: Dict[str, Any]) -> List[ModalityAgent]:
        """Create modality agents from a single sample."""
        agents = []
        
        for modality_result in sample['modality_results']:
            modality = modality_result['modality']
            name_map = {
                'T': 'Text',
                'A': 'Audio', 
                'V': 'Visual',
                'TAV': 'Multimodal'
            }
            
            agent = ModalityAgent(
                name=name_map.get(modality, modality),
                modality=modality,
                primary_emotion=modality_result['primary_emotion'],
                primary_confidence=modality_result['primary_confidence'],
                data_quality_score=modality_result['data_quality']['score'],
                data_quality_issues=modality_result['data_quality']['issues'],
                reasoning=modality_result['reasoning'],
                emotions=modality_result['emotions']
            )
            agents.append(agent)
        
        return agents
    
    def conduct_discussion(self, agents: List[ModalityAgent], video_id: str, ground_truth: str) -> Dict[str, Any]:
        """Conduct a discussion between modality agents."""
        discussion_log = []
        discussion_log.append(f"=== DISCUSSION FOR {video_id} (Ground Truth: {ground_truth}) ===")
        discussion_log.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        discussion_log.append("")
        
        # Phase 1: Introduction
        discussion_log.append("PHASE 1: AGENT INTRODUCTIONS")
        discussion_log.append("-" * 50)
        for agent in agents:
            discussion_log.append(agent.introduce_self())
        discussion_log.append("")
        
        # Phase 2: Confidence Discussion
        discussion_log.append("PHASE 2: CONFIDENCE DISCUSSION")
        discussion_log.append("-" * 50)
        for agent in agents:
            other_agents = [a for a in agents if a != agent]
            discussion_log.append(agent.discuss_confidence(other_agents))
        discussion_log.append("")
        
        # Phase 3: Data Quality Discussion
        discussion_log.append("PHASE 3: DATA QUALITY ASSESSMENT")
        discussion_log.append("-" * 50)
        for agent in agents:
            discussion_log.append(agent.discuss_data_quality())
        discussion_log.append("")
        
        # Phase 4: Disagreement Resolution
        discussion_log.append("PHASE 4: DISAGREEMENT RESOLUTION")
        discussion_log.append("-" * 50)
        emotions = [agent.primary_emotion for agent in agents]
        unique_emotions = list(set(emotions))
        
        if len(unique_emotions) > 1:
            # Find agents with different emotions
            for i, agent1 in enumerate(agents):
                for agent2 in agents[i+1:]:
                    if agent1.primary_emotion != agent2.primary_emotion:
                        discussion_log.append(agent1.respond_to_disagreement(agent2))
                        discussion_log.append(agent2.respond_to_disagreement(agent1))
        else:
            discussion_log.append("All agents agree on the primary emotion!")
        discussion_log.append("")
        
        # Phase 5: Consensus Building
        discussion_log.append("PHASE 5: CONSENSUS BUILDING")
        discussion_log.append("-" * 50)
        for agent in agents:
            discussion_log.append(agent.propose_consensus(agents))
        discussion_log.append("")
        
        # Phase 6: Final Conclusion
        discussion_log.append("PHASE 6: FINAL CONCLUSION")
        discussion_log.append("-" * 50)
        
        # Calculate weighted consensus
        emotion_weights = {}
        for agent in agents:
            emotion = agent.primary_emotion
            weight = (agent.primary_confidence / 100) * (agent.data_quality_score / 100)
            emotion_weights[emotion] = emotion_weights.get(emotion, 0) + weight
        
        final_emotion = max(emotion_weights.items(), key=lambda x: x[1])
        final_confidence = final_emotion[1] / len(agents) * 100
        
        discussion_log.append(f"FINAL CONSENSUS: '{final_emotion[0]}' with {final_confidence:.1f}% confidence")
        discussion_log.append(f"Ground Truth: '{ground_truth}'")
        discussion_log.append(f"Consensus Correct: {final_emotion[0] == ground_truth}")
        discussion_log.append("")
        
        return {
            'video_id': video_id,
            'discussion_log': discussion_log,
            'final_emotion': final_emotion[0],
            'final_confidence': final_confidence,
            'ground_truth': ground_truth,
            'consensus_correct': final_emotion[0] == ground_truth,
            'emotion_weights': emotion_weights
        }
    
    def run_discussions(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """Run discussions for all samples or up to max_samples."""
        results = []
        samples_to_process = self.data[:max_samples] if max_samples else self.data
        
        print(f"Running discussions for {len(samples_to_process)} samples...")
        
        for i, sample in enumerate(samples_to_process):
            print(f"Processing sample {i+1}/{len(samples_to_process)}: {sample['video_id']}")
            
            agents = self.create_agents_from_sample(sample)
            result = self.conduct_discussion(
                agents, 
                sample['video_id'], 
                sample['ground_truth']
            )
            results.append(result)
        
        return results
    
    def save_discussions(self, results: List[Dict[str, Any]], output_file: str):
        """Save discussion results to a file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                for line in result['discussion_log']:
                    f.write(line + '\n')
                f.write('\n' + '='*80 + '\n\n')
        
        print(f"Discussions saved to {output_file}")
    
    def compare_with_original_ensemble(self, results: List[Dict[str, Any]]) -> str:
        """Compare discussion results with original ensemble results."""
        comparison_report = []
        comparison_report.append("COMPARISON WITH ORIGINAL ENSEMBLE RESULTS")
        comparison_report.append("=" * 50)
        
        discussion_correct = 0
        ensemble_correct = 0
        agreement_count = 0
        
        for i, result in enumerate(results):
            video_id = result.get('video_id', f'sample_{i}')
            discussion_prediction = result['final_emotion']
            ground_truth = result['ground_truth']
            
            # Get original ensemble prediction from the data
            original_sample = self.data[i]
            ensemble_prediction = original_sample['ensemble_prediction']
            ensemble_confidence = original_sample['ensemble_confidence']
            
            discussion_correct += (discussion_prediction == ground_truth)
            ensemble_correct += (ensemble_prediction == ground_truth)
            agreement_count += (discussion_prediction == ensemble_prediction)
            
            comparison_report.append(f"\nVideo: {video_id}")
            comparison_report.append(f"  Ground Truth: {ground_truth}")
            comparison_report.append(f"  Discussion Prediction: {discussion_prediction} ({result['final_confidence']:.1f}%)")
            comparison_report.append(f"  Ensemble Prediction: {ensemble_prediction} ({ensemble_confidence:.1f}%)")
            comparison_report.append(f"  Discussion Correct: {discussion_prediction == ground_truth}")
            comparison_report.append(f"  Ensemble Correct: {ensemble_prediction == ground_truth}")
            comparison_report.append(f"  Predictions Agree: {discussion_prediction == ensemble_prediction}")
        
        total_samples = len(results)
        discussion_accuracy = (discussion_correct / total_samples) * 100
        ensemble_accuracy = (ensemble_correct / total_samples) * 100
        agreement_rate = (agreement_count / total_samples) * 100
        
        comparison_report.append(f"\nSUMMARY COMPARISON:")
        comparison_report.append(f"  Discussion Accuracy: {discussion_accuracy:.2f}%")
        comparison_report.append(f"  Ensemble Accuracy: {ensemble_accuracy:.2f}%")
        comparison_report.append(f"  Accuracy Difference: {discussion_accuracy - ensemble_accuracy:.2f}%")
        comparison_report.append(f"  Prediction Agreement Rate: {agreement_rate:.2f}%")
        
        return "\n".join(comparison_report)

    def generate_summary_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a summary report of all discussions."""
        total_samples = len(results)
        correct_predictions = sum(1 for r in results if r['consensus_correct'])
        accuracy = (correct_predictions / total_samples) * 100
        
        # Calculate per-emotion accuracy
        emotion_accuracy = {}
        emotion_counts = {}
        emotion_correct = {}
        
        for result in results:
            predicted = result['final_emotion']
            ground_truth = result['ground_truth']
            
            emotion_counts[predicted] = emotion_counts.get(predicted, 0) + 1
            if predicted == ground_truth:
                emotion_correct[predicted] = emotion_correct.get(predicted, 0) + 1
        
        for emotion in emotion_counts:
            if emotion in emotion_correct:
                emotion_accuracy[emotion] = (emotion_correct[emotion] / emotion_counts[emotion]) * 100
            else:
                emotion_accuracy[emotion] = 0.0
        
        # Calculate confusion matrix
        confusion_matrix = {}
        for result in results:
            predicted = result['final_emotion']
            ground_truth = result['ground_truth']
            
            if ground_truth not in confusion_matrix:
                confusion_matrix[ground_truth] = {}
            if predicted not in confusion_matrix[ground_truth]:
                confusion_matrix[ground_truth][predicted] = 0
            confusion_matrix[ground_truth][predicted] += 1
        
        # Calculate average confidence for correct vs incorrect predictions
        correct_confidences = [r['final_confidence'] for r in results if r['consensus_correct']]
        incorrect_confidences = [r['final_confidence'] for r in results if not r['consensus_correct']]
        
        avg_correct_confidence = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
        avg_incorrect_confidence = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0
        
        report = f"""
DETAILED ACCURACY ANALYSIS
=========================
Total Samples: {total_samples}
Correct Predictions: {correct_predictions}
Overall Accuracy: {accuracy:.2f}%

Per-Emotion Accuracy:
"""
        for emotion in sorted(emotion_accuracy.keys()):
            report += f"  {emotion}: {emotion_accuracy[emotion]:.1f}% ({emotion_correct.get(emotion, 0)}/{emotion_counts[emotion]} correct)\n"
        
        report += f"""
Confidence Analysis:
  Average Confidence (Correct Predictions): {avg_correct_confidence:.1f}%
  Average Confidence (Incorrect Predictions): {avg_incorrect_confidence:.1f}%
  Confidence Difference: {avg_correct_confidence - avg_incorrect_confidence:.1f}%

Confusion Matrix:
"""
        for ground_truth in sorted(confusion_matrix.keys()):
            report += f"  Ground Truth '{ground_truth}':\n"
            for predicted in sorted(confusion_matrix[ground_truth].keys()):
                count = confusion_matrix[ground_truth][predicted]
                report += f"    Predicted '{predicted}': {count}\n"
        
        return report

def main():
    """Main function to run the modality discussion system."""
    # Initialize the system
    json_file = "final/result/MER/mar-workshop/all-emos-data-quality_test_ensemble_gpt5-nano.json"
    discussion_system = ModalityDiscussionSystem(json_file)
    
    # Run discussions for first 5 samples (you can change this number)
    results = discussion_system.run_discussions(max_samples=5)
    
    # Save detailed discussions
    discussion_system.save_discussions(results, "modality_discussions.txt")
    
    # Generate and print summary report
    summary = discussion_system.generate_summary_report(results)
    print(summary)
    
    # Generate comparison with original ensemble
    comparison = discussion_system.compare_with_original_ensemble(results)
    print("\n" + comparison)
    
    # Save summary and comparison to files
    with open("discussion_summary.txt", "w") as f:
        f.write(summary)
        f.write("\n\n" + comparison)
    
    print("\nAnalysis complete! Check 'modality_discussions.txt' for detailed discussions and 'discussion_summary.txt' for summary and comparison.")

if __name__ == "__main__":
    main()
