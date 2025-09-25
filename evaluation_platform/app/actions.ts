"use server"

import { revalidatePath } from 'next/cache'
import { redirect } from 'next/navigation'
import { createClient } from '@/app/utils/supabase/server'

export async function signout() {
  const supabase = await createClient()
  const { error } = await supabase.auth.signOut()
  if (error) redirect('/error')
  redirect('/login')
}

export async function getUserInfo() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  const { data, error } = await supabase
    .from("users")
    .select("id, set_id")
    .eq("auth_uuid", user?.id)
    .single();

  if (error) {
    console.error("Error fetching user info:", error);
    return null;
  }
  return data;
}

export async function getEvaluation(setNumber: number, userId: number) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) throw new Error("User not authenticated");

  const { data, error } = await supabase
    .from("evaluations")
    .select(
      `
      indication_1_comprehensiveness,
      indication_2_comprehensiveness,
      indication_3_comprehensiveness,
      indication_4_comprehensiveness,
      indication_1_factuality,
      indication_2_factuality,
      indication_3_factuality,
      indication_4_factuality,
      indication_1_conciseness,
      indication_2_conciseness,
      indication_3_conciseness,
      indication_4_conciseness,
      indication_1_comment,
      indication_2_comment,
      indication_3_comment,
      indication_4_comment,
      protocol_usefulness_ranking,
      interpretation_usefulness_ranking,
      overall_ranking,
      factor_ranking,
      general_comment
      `
    )
    .eq("user_id", userId)
    .eq("set_id", setNumber)
    .single();

  if (error) {
    if (error.code === 'PGRST116') {
      // No row found for this user and setNumber
      return null;
    }
    throw error;
  }

  return {
    indications: {
      indication1: {
        comprehensiveness: data.indication_1_comprehensiveness,
        factuality: data.indication_1_factuality,
        conciseness: data.indication_1_conciseness,
      },
      indication2: {
        comprehensiveness: data.indication_2_comprehensiveness,
        factuality: data.indication_2_factuality,
        conciseness: data.indication_2_conciseness,
      },
      indication3: {
        comprehensiveness: data.indication_3_comprehensiveness,
        factuality: data.indication_3_factuality,
        conciseness: data.indication_3_conciseness,
      },
      indication4: {
        comprehensiveness: data.indication_4_comprehensiveness,
        factuality: data.indication_4_factuality,
        conciseness: data.indication_4_conciseness,
      }
    },
    comments: {
      indication1: data.indication_1_comment || '',
      indication2: data.indication_2_comment || '',
      indication3: data.indication_3_comment || '',
      indication4: data.indication_4_comment || ''
    },
    overallRanking: data.overall_ranking,
    protocolRanking: data.protocol_usefulness_ranking,
    interpretationRanking: data.interpretation_usefulness_ranking,
    factorRanking: data.factor_ranking,
    generalComment: data.general_comment || ''
  };
}

export async function submitData(
  comprehensiveness1: number,
  comprehensiveness2: number,
  comprehensiveness3: number,
  comprehensiveness4: number,
  factuality1: number,
  factuality2: number,
  factuality3: number,
  factuality4: number,
  conciseness1: number,
  conciseness2: number,
  conciseness3: number,
  conciseness4: number,
  comment1: string,
  comment2: string,
  comment3: string,
  comment4: string,
  protocolRanking: string,
  interpretationRanking: string,
  overallRanking: string,
  factorRanking: string,
  generalComment: string,
  setNumber: number
) {

  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  const { data, error } = await supabase
    .from("users")
    .select("id, set_id")
    .eq("auth_uuid", user?.id)
  const user_id = data?.[0].id

  const { error: upsertError } = await supabase
    .from('evaluations')
    .upsert({
      user_id: user_id,
      set_id: setNumber,
      indication_1_comprehensiveness: comprehensiveness1,
      indication_2_comprehensiveness: comprehensiveness2,
      indication_3_comprehensiveness: comprehensiveness3,
      indication_4_comprehensiveness: comprehensiveness4,
      indication_1_factuality: factuality1,
      indication_2_factuality: factuality2,
      indication_3_factuality: factuality3,
      indication_4_factuality: factuality4,
      indication_1_conciseness: conciseness1,
      indication_2_conciseness: conciseness2,
      indication_3_conciseness: conciseness3,
      indication_4_conciseness: conciseness4,
      indication_1_comment: comment1,
      indication_2_comment: comment2,
      indication_3_comment: comment3,
      indication_4_comment: comment4,
      protocol_usefulness_ranking: protocolRanking,
      interpretation_usefulness_ranking: interpretationRanking,
      overall_ranking: overallRanking,
      factor_ranking: factorRanking,
      general_comment: generalComment
    }, {
      onConflict: 'user_id,set_id' // ensure your DB has a unique constraint on (user_id, set_id)
    });

  if (upsertError) {
    throw upsertError;
  }

  // Update current set only if user is on latest one
  const currentSetId = data?.[0].set_id || 1;
  if (setNumber === currentSetId) {
    await supabase
      .from('users')
      .update({ set_id: setNumber + 1 })
      .eq('id', user_id);
  }
}
