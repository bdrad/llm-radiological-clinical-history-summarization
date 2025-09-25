"use client";

import React, { useEffect, useState } from 'react';
import Slider from '@/components/Slider';
import { redirect } from 'next/navigation';
import { getEvaluation, getUserInfo, signout, submitData } from "./actions";

type IndicationScore = {
  comprehensiveness: number;
  factuality: number;
  conciseness: number;
};

type IndicationScores = {
  indication1: IndicationScore;
  indication2: IndicationScore;
  indication3: IndicationScore;
  indication4: IndicationScore;
};

type IndicationTexts = {
  indication1: string;
  indication2: string;
  indication3: string;
  indication4: string;
};

type Comments = {
  indication1: string;
  indication2: string;
  indication3: string;
  indication4: string;
};

type Note = {
  title: string;
  text: string;
};

type Notes = {
  [key: string]: Note;
};

type Ranking = {
  [option: string]: number | null;
};

interface RankingTableProps {
  title: string;
  options: string[];
  ranking: Ranking;
  setRanking: React.Dispatch<React.SetStateAction<Ranking>>;
  tableName: string;
  showValidation: boolean;
}

function RankingTable({ title, options, ranking, setRanking, tableName, showValidation }: RankingTableProps) {
  const handleSelection = (option: string, rank: number) => {
    setRanking(prev => {
      const newRanking = { ...prev };
      Object.keys(newRanking).forEach(key => {
        if (key !== option && newRanking[key] === rank) {
          newRanking[key] = null;
        }
      });
      newRanking[option] = rank;
      return newRanking;
    });
  };

  const isRankingComplete = Object.values(ranking).every(rank => rank !== null);

  return (
    <div className="mb-6">
      <p className="font-medium mb-2">{title}</p>
      <table className="w-full mt-2 border-collapse">
        <thead>
          <tr>
            <th className="p-2 border text-center">Rank</th>
            {options.map((option, idx) => (
              <th key={idx} className="p-2 border text-center">{option}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: options.length }, (_, i) => i + 1).map(rank => (
            <tr key={rank}>
              <td className="p-2 border text-center font-medium">{rank}</td>
              {options.map((option, idx) => (
                <td key={idx} className="p-2 border text-center">
                  <input
                    type="radio"
                    name={`${tableName}-${option}`} 
                    value={rank}
                    checked={ranking[option] === rank}
                    onChange={() => handleSelection(option, rank)}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {showValidation && !isRankingComplete && (
        <p className="text-red-500 text-sm mt-2">
          ⚠️ Please assign a rank to every option above.
        </p>
      )}
    </div>
  );
}

export default function Home() {
  const TOTAL_CASES = 25;

  const [setNumber, setSetNumber] = useState(1);
  const [userId, setUserId] = useState(1);
  const [maxAvailableSet, setMaxAvailableSet] = useState(1); // highest case user is allowed to view

  const [exam, setExam] = useState("");
  const [showNotes, setShowNotes] = useState(true);
  const [generalComment, setGeneralComment] = useState('');
  const [showValidation, setShowValidation] = useState(false);

  const [indicationScores, setIndicationScores] = useState<IndicationScores>({
    indication1: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
    indication2: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
    indication3: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
    indication4: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
  });

  const [indicationTexts, setIndicationTexts] = useState<IndicationTexts>({
    indication1: "",
    indication2: "",
    indication3: "",
    indication4: "",
  });

  const [comments, setComments] = useState<Comments>({
    indication1: '',
    indication2: '',
    indication3: '',
    indication4: ''
  });

  const [notes, setNotes] = useState<Notes>({
    note1: { title: "", text: "" },
    note2: { title: "", text: "" },
    note3: { title: "", text: "" },
    note4: { title: "", text: "" },
    note5: { title: "", text: "" },
    note6: { title: "", text: "" },
    note7: { title: "", text: "" },
    note8: { title: "", text: "" },
    note9: { title: "", text: "" },
    note10: { title: "", text: "" }
  });

  const [protocolRanking, setProtocolRanking] = useState<Ranking>({
    "Indication 1": null,
    "Indication 2": null,
    "Indication 3": null,
    "Indication 4": null,
  });
  const [interpretationRanking, setInterpretationRanking] = useState<Ranking>({
    "Indication 1": null,
    "Indication 2": null,
    "Indication 3": null,
    "Indication 4": null,
  });
  const [overallRanking, setOverallRanking] = useState<Ranking>({
    "Indication 1": null,
    "Indication 2": null,
    "Indication 3": null,
    "Indication 4": null,
  });
  const [factorRanking, setFactorRanking] = useState<Ranking>({
    "Comprehensiveness": null,
    "Factuality": null,
    "Conciseness": null,
  });

  const handleSliderChange = (indication: keyof IndicationScores, metric: keyof IndicationScore, value: number) => {
    setIndicationScores(prev => ({
      ...prev,
      [indication]: { ...prev[indication], [metric]: value },
    }));
  };

  const handleCommentChange = (indication: keyof Comments, value: string) => {
    setComments(prev => ({
      ...prev,
      [indication]: value,
    }));
  };

  const handleSignout = async () => {
    await signout();
  };

  const handleSubmitData = async () => {
    const isProtocolRankingComplete = Object.values(protocolRanking).every(rank => rank !== null);
    const isInterpretationRankingComplete = Object.values(interpretationRanking).every(rank => rank !== null);
    const isOverallRankingComplete = Object.values(overallRanking).every(rank => rank !== null);
    const isFactorRankingComplete = Object.values(factorRanking).every(rank => rank !== null);

    if (!isProtocolRankingComplete || !isInterpretationRankingComplete || !isOverallRankingComplete || !isFactorRankingComplete) {
      setShowValidation(true);
      return;
    }

    setShowValidation(false);

    await submitData(
      indicationScores.indication1.comprehensiveness,
      indicationScores.indication2.comprehensiveness,
      indicationScores.indication3.comprehensiveness,
      indicationScores.indication4.comprehensiveness,
      indicationScores.indication1.factuality,
      indicationScores.indication2.factuality,
      indicationScores.indication3.factuality,
      indicationScores.indication4.factuality,
      indicationScores.indication1.conciseness,
      indicationScores.indication2.conciseness,
      indicationScores.indication3.conciseness,
      indicationScores.indication4.conciseness,
      comments.indication1,
      comments.indication2,
      comments.indication3,
      comments.indication4,
      JSON.stringify(protocolRanking),
      JSON.stringify(interpretationRanking),
      JSON.stringify(overallRanking),
      JSON.stringify(factorRanking),
      generalComment,
      setNumber
    );

    // Reset states for next case
    setIndicationScores({
      indication1: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
      indication2: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
      indication3: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
      indication4: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
    });
    setComments({ indication1: '', indication2: '', indication3: '', indication4: '' });
    setProtocolRanking({
      "Indication 1": null,
      "Indication 2": null,
      "Indication 3": null,
      "Indication 4": null,
    });
    setInterpretationRanking({
      "Indication 1": null,
      "Indication 2": null,
      "Indication 3": null,
      "Indication 4": null,
    });
    setOverallRanking({
      "Indication 1": null,
      "Indication 2": null,
      "Indication 3": null,
      "Indication 4": null,
    });
    setFactorRanking({
      "Comprehensiveness": null,
      "Factuality": null,
      "Conciseness": null,
    });

    setSetNumber(prev => prev + 1);
    if (setNumber === maxAvailableSet) {
      setMaxAvailableSet(prev => Math.max(prev + 1, maxAvailableSet));
    }
    window.scrollTo({ top: 0 });
  };

  useEffect(() => {
    const initializeUserInfo = async () => {
      const userInfo = await getUserInfo();
      if (userInfo) {
        setMaxAvailableSet(userInfo.set_id || 1);
        setSetNumber(userInfo.set_id || 1);
        setUserId(userInfo.id || 1);
      } else {
        console.error('Failed to get user info');
        setMaxAvailableSet(1);
        setSetNumber(1);
        setUserId(1);
      }
    };
    initializeUserInfo();
  }, []); 

  useEffect(() => {
    const fetchData = async (setNumber: number, userId: number) => {
      const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || (process.env.NODE_ENV === 'production'
        ? 'https://your-deployment-url.com'
        : 'http://localhost:3000');
  
      try {
        const examResponse = await fetch(`${baseUrl}/user${userId}/set${setNumber}/exam.txt`);
        setExam(await examResponse.text());
      } catch (error) {
        console.error('Error fetching exam:', error);
      }
  
      try {
        const responses = await Promise.all(
          [1, 2, 3, 4].map(i =>
            fetch(`${baseUrl}/user${userId}/set${setNumber}/indication${i}.txt`).then(res => res.text())
          )
        );
        setIndicationTexts({
          indication1: responses[0],
          indication2: responses[1],
          indication3: responses[2],
          indication4: responses[3],
        });
      } catch (error) {
        console.error('Error fetching indications:', error);
      }
  
      try {
        const responses = await Promise.all(
          Array.from({ length: 10 }, (_, i) =>
            fetch(`${baseUrl}/user${userId}/set${setNumber}/note${i + 1}.txt`).then(res => res.text())
          )
        );
        const parsed = responses.map(text => {
          const [title, ...body] = text.split('\n');
          return { title, text: body.join('\n') };
        });
        setNotes(Object.fromEntries(parsed.map((n, i) => [`note${i + 1}`, n])));
      } catch (error) {
        console.error('Error fetching notes:', error);
      }
  
      try {
        const saved = await getEvaluation(setNumber, userId);
        if (saved) {
          setIndicationScores(prev => ({
            indication1: { ...prev.indication1, ...saved.indications.indication1 },
            indication2: { ...prev.indication2, ...saved.indications.indication2 },
            indication3: { ...prev.indication3, ...saved.indications.indication3 },
            indication4: { ...prev.indication4, ...saved.indications.indication4 },
          }));
          setComments(saved.comments);
          setProtocolRanking(JSON.parse(saved.protocolRanking));
          setInterpretationRanking(JSON.parse(saved.interpretationRanking));
          setOverallRanking(JSON.parse(saved.overallRanking));
          setFactorRanking(JSON.parse(saved.factorRanking));
          setGeneralComment(saved.generalComment);
        } else {
          setIndicationScores({
            indication1: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
            indication2: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
            indication3: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
            indication4: { comprehensiveness: 5, factuality: 5, conciseness: 5 },
          });
          setComments({
            indication1: '',
            indication2: '',
            indication3: '',
            indication4: '',
          });
          setProtocolRanking({
            "Indication 1": null,
            "Indication 2": null,
            "Indication 3": null,
            "Indication 4": null,
          });
          setInterpretationRanking({
            "Indication 1": null,
            "Indication 2": null,
            "Indication 3": null,
            "Indication 4": null,
          });
          setOverallRanking({
            "Indication 1": null,
            "Indication 2": null,
            "Indication 3": null,
            "Indication 4": null,
          });
          setFactorRanking({
            "Comprehensiveness": null,
            "Factuality": null,
            "Conciseness": null,
          });
        }
      } catch (error) {
        console.error('Error fetching saved evaluation:', error);
      }
    };
  
    fetchData(setNumber, userId);
  }, [setNumber, userId]);

  if (setNumber > TOTAL_CASES) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100 p-5">
        <div className="w-full max-w-4xl bg-white shadow-lg rounded-lg p-8 text-center">
          <h2 className="text-2xl font-semibold mb-6">Thank you for participating!</h2>
          <div className="flex justify-center gap-4">
            <button
              onClick={() => setSetNumber(prev => Math.max(prev - 1, 1))}
              className="px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg"
            >
              Previous Case
            </button>
            <button
              onClick={handleSignout}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Sign Out
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-5">
      <div className="w-full max-w-8xl bg-white shadow-lg rounded-lg p-8 flex flex-col">
        <div className="relative w-full flex items-center justify-center mb-6">
          <h2 className="px-4 py-2 text-lg rounded-lg text-gray">
            Case {setNumber} of {TOTAL_CASES}
          </h2>
          <button
            onClick={handleSignout}
            className="absolute right-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 min-w-[100px] text-center"
          >
            Sign Out
          </button>
          <button
            onClick={() => setShowNotes(prev => !prev)}
            className="absolute right-[120px] px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg"
          >
            {showNotes ? 'Hide Notes' : 'Show Notes'}
          </button>
          <div className="absolute left-4 flex gap-4">
            <button
              onClick={() => setSetNumber(prev => Math.max(prev - 1, 1))}
              disabled={setNumber <= 1}
              className={`px-4 py-2 rounded-lg text-gray-800 ${
                setNumber <= 1
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-gray-200 hover:bg-gray-300'
              }`}
            >
              Previous Case
            </button>
            <button
              onClick={() => setSetNumber(prev => Math.min(prev + 1, TOTAL_CASES + 1))}
              disabled={setNumber >= maxAvailableSet}
              className={`px-4 py-2 rounded-lg text-gray-800 ${
                setNumber >= maxAvailableSet
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-gray-200 hover:bg-gray-300'
              }`}
            >
              Next Case
            </button>
          </div>
        </div>
        <div className="flex flex-row">
          {showNotes && (
            <div className="w-1/2 pr-4">
              <h2 className="text-xl font-semibold mb-4">Notes</h2>
              <div className="space-y-4">
                {Object.entries(notes).map(([key, note]) => (
                  <details key={key} className="bg-gray-100 p-4 rounded-lg">
                    <summary className="cursor-pointer text-lg font-medium">{note.title}</summary>
                    <p className="mt-2 text-sm text-gray-600 overflow-y-auto whitespace-pre-wrap" style={{ height: '600px' }}>
                      {note.text}
                    </p>
                  </details>
                ))}
              </div>
            </div>
          )}
          <div className={`${showNotes ? 'w-1/2' : 'w-full'} pl-4`}>
            <h2 className="text-xl font-semibold mb-4">Evaluation</h2>
            <div className="bg-gray-100 p-4 rounded-lg space-y-6">
              <div>
                <h3 className="text-lg font-medium">Exam</h3>
                <p className="text-md">{exam}</p>
              </div>
              <div className={`grid gap-4 ${showNotes ? 'grid-cols-2' : 'grid-cols-4'}`}>
                {Object.entries(indicationScores).map(([key, val]) => (
                  <div key={key} className="bg-white rounded p-4 shadow">
                    <h3 className="text-lg font-medium capitalize">{key.replace('indication', 'Indication ')}</h3>
                    <p className="text-sm mt-2 mb-2 max-h-48 overflow-y-auto">{indicationTexts[key as keyof IndicationTexts]}</p>
                    <Slider label="Comprehensiveness" value={val.comprehensiveness} onChange={(v) => handleSliderChange(key as keyof IndicationScores, 'comprehensiveness', v)} />
                    <Slider label="Factuality" value={val.factuality} onChange={(v) => handleSliderChange(key as keyof IndicationScores, 'factuality', v)} />
                    <Slider label="Conciseness" value={val.conciseness} onChange={(v) => handleSliderChange(key as keyof IndicationScores, 'conciseness', v)} />
                    <textarea
                      placeholder="Comment (Optional)"
                      className="mt-2 w-full p-2 border border-gray-300 rounded-lg"
                      value={comments[key as keyof Comments]}
                      onChange={(e) => handleCommentChange(key as keyof Comments, e.target.value)}
                    />
                  </div>
                ))}
              </div>

              {/* Ranking UIs for each question */}
              <RankingTable
                title="Rank the indications based on which would be most useful for protocoling this exam (1 = Most useful, 4 = Least useful):"
                options={["Indication 1", "Indication 2", "Indication 3", "Indication 4"]}
                ranking={protocolRanking}
                setRanking={setProtocolRanking}
                tableName="protocol"
                showValidation={showValidation}
              />

              <RankingTable
                title="Rank the indications based on which would be most useful to have while interpreting this exam (1 = Most useful, 4 = Least useful):"
                options={["Indication 1", "Indication 2", "Indication 3", "Indication 4"]}
                ranking={interpretationRanking}
                setRanking={setInterpretationRanking}
                tableName="interpretation"
                showValidation={showValidation}
              />

              <RankingTable
                title="Rank the indications based on your overall preference (1 = Most preferred, 4 = Least preferred):"
                options={["Indication 1", "Indication 2", "Indication 3", "Indication 4"]}
                ranking={overallRanking}
                setRanking={setOverallRanking}
                tableName="overall"
                showValidation={showValidation}
              />

              <RankingTable
                title="Rank the factors based on how much they influenced your overall preference (1 = Most influential, 3 = Least influential):"
                options={["Comprehensiveness", "Factuality", "Conciseness"]}
                ranking={factorRanking}
                setRanking={setFactorRanking}
                tableName="factor"
                showValidation={showValidation}
              />

              <div>
                <p className="font-medium">Any general comments about this case?</p>
                <textarea
                  placeholder="General feedback (Optional)"
                  value={generalComment}
                  onChange={(e) => setGeneralComment(e.target.value)}
                  className="mt-2 w-full p-2 border border-gray-300 rounded-lg"
                  rows={2}
                />
              </div>

              <div className="flex justify-center">
                <button
                  onClick={handleSubmitData}
                  className="px-4 py-2 bg-blue-700 text-white rounded-lg hover:bg-blue-800"
                >
                  Submit Evaluation
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
